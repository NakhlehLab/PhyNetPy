"""
Automated deployment script for PhyNetPy.

Usage:
    python deploy.py patch     # 0.3.0 -> 0.3.1
    python deploy.py minor     # 0.3.0 -> 0.4.0
    python deploy.py major     # 0.3.0 -> 1.0.0
    python deploy.py --dry-run --no-bump  # Validate the prepared version
    python deploy.py --no-bump            # Upload the prepared version
    python deploy.py --full-tests --no-bump  # Include slow tests

Requires:
    - pytest, build, twine, check-manifest
    - A PyPI API token stored in .pypi_token (one line, no whitespace)
"""

import argparse
import os
import subprocess
import sys
import re
from pathlib import Path

VERSION_FILE = Path(__file__).parent / "src" / "_version.py"
TOKEN_FILE = Path(__file__).parent / ".pypi_token"
DIST_DIR = Path(__file__).parent / "dist"

VERSION_RE = r'(__version__\s*=\s*")(\d+\.\d+\.\d+)(")'


def read_current_version() -> str:
    text = VERSION_FILE.read_text(encoding="utf-8")
    match = re.search(VERSION_RE, text)
    if not match:
        sys.exit(f"Could not find __version__ in {VERSION_FILE}")
    return match.group(2)


def bump_version(current: str, bump_type: str) -> str:
    major, minor, patch = (int(x) for x in current.split("."))
    if bump_type == "major":
        return f"{major + 1}.0.0"
    elif bump_type == "minor":
        return f"{major}.{minor + 1}.0"
    elif bump_type == "patch":
        return f"{major}.{minor}.{patch + 1}"
    else:
        sys.exit(f"Invalid bump type: {bump_type!r}. Use 'major', 'minor', or 'patch'.")


def write_version(new_version: str) -> None:
    text = VERSION_FILE.read_text(encoding="utf-8")
    updated = re.sub(VERSION_RE, rf"\g<1>{new_version}\g<3>", text)
    VERSION_FILE.write_text(updated, encoding="utf-8")


def release_checkout_ready() -> bool:
    """Require a clean ``main`` checkout before an actual upload."""
    branch = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
    )
    if branch.returncode != 0 or status.returncode != 0:
        print("Could not inspect the Git checkout.")
        return False
    if branch.stdout.strip() != "main":
        print("PyPI uploads must be made from the main branch.")
        return False
    if status.stdout.strip():
        print("PyPI uploads require a clean working tree.")
        return False
    return True


def run_tests() -> bool:
    print("\n========== Running tests ==========\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v", "-m", "not slow"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def run_full_tests() -> bool:
    print("\n========== Running full test suite ==========\n")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-v"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def check_manifest() -> bool:
    print("\n========== Checking source manifest ==========\n")
    result = subprocess.run(
        [sys.executable, "-m", "check_manifest"],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def clean_dist() -> None:
    if DIST_DIR.exists():
        import shutil
        shutil.rmtree(DIST_DIR)


def build_package() -> bool:
    print("\n========== Building package ==========\n")
    result = subprocess.run([sys.executable, "-m", "build"], cwd=Path(__file__).parent)
    return result.returncode == 0


def distribution_files() -> list[str]:
    """Return built wheel and source-distribution paths in stable order."""
    return [
        str(path)
        for path in sorted(DIST_DIR.iterdir())
        if path.is_file() and (path.suffix == ".whl" or path.name.endswith(".tar.gz"))
    ]


def check_distribution() -> bool:
    print("\n========== Checking built distributions ==========\n")
    artifacts = distribution_files()
    if not artifacts:
        print("No wheel or source distribution found in dist/.")
        return False
    result = subprocess.run(
        [sys.executable, "-m", "twine", "check", *artifacts],
        cwd=Path(__file__).parent,
    )
    return result.returncode == 0


def upload_to_pypi(token: str) -> bool:
    print("\n========== Uploading to PyPI ==========\n")
    env = dict(os.environ)
    env["TWINE_USERNAME"] = "__token__"
    env["TWINE_PASSWORD"] = token
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            *distribution_files(),
        ],
        cwd=Path(__file__).parent,
        env=env,
    )
    return result.returncode == 0


def load_token() -> str:
    if not TOKEN_FILE.exists():
        sys.exit(
            f"PyPI token file not found at {TOKEN_FILE}\n"
            "Create it with your API token (single line, no whitespace):\n"
            f"  echo pypi-YOUR-TOKEN-HERE > {TOKEN_FILE}"
        )
    token = TOKEN_FILE.read_text(encoding="utf-8").strip()
    if not token:
        sys.exit("Token file is empty.")
    return token


def parse_args() -> argparse.Namespace:
    """Parse release validation and deployment arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "bump",
        nargs="?",
        choices=("major", "minor", "patch"),
        help="semantic version component to increment",
    )
    parser.add_argument(
        "--no-bump",
        action="store_true",
        help="validate or upload the version already in src/_version.py",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run checks and build artifacts without uploading",
    )
    parser.add_argument(
        "--full-tests",
        action="store_true",
        help="include slow recovery and stress tests",
    )
    args = parser.parse_args()
    if args.no_bump == (args.bump is not None):
        parser.error("choose exactly one of a version bump or --no-bump")
    return args


def main() -> None:
    args = parse_args()
    current = read_current_version()
    new_version = current if args.no_bump else bump_version(current, args.bump)

    print(f"Current version : {current}")
    print(f"Bump type       : {args.bump or 'none (prepared release)'}")
    print(f"New version     : {new_version}")

    if not args.dry_run:
        if not args.no_bump:
            sys.exit(
                "\nActual uploads require a committed release version; "
                "prepare it first, then use --no-bump."
            )
        if not release_checkout_ready():
            sys.exit("\nRelease checkout is not ready — aborting deployment.")
        token = load_token()

    tests_ok = run_full_tests() if args.full_tests else run_tests()
    if not tests_ok:
        sys.exit("\nTests failed — aborting deployment.")

    if not check_manifest():
        sys.exit("\nSource manifest check failed — aborting deployment.")

    version_changed = new_version != current
    if version_changed:
        write_version(new_version)
        print(f"\nVersion updated: {current} -> {new_version}")

    clean_dist()

    if not build_package():
        if version_changed:
            write_version(current)
        sys.exit("\nBuild failed — version rolled back.")

    if not check_distribution():
        if version_changed:
            write_version(current)
        sys.exit("\nDistribution validation failed — version rolled back.")

    if args.dry_run:
        print("\n[DRY RUN] Skipping PyPI upload.")
        print(f"Build artifacts are in {DIST_DIR}")
        return

    if not upload_to_pypi(token):
        sys.exit("\nUpload to PyPI failed. Built artifacts remain in dist/.")

    print(f"\nSuccessfully deployed phynetpy {new_version} to PyPI!")


if __name__ == "__main__":
    main()
