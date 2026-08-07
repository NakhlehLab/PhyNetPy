"""
Automated deployment script for PhyNetPy.

Usage:
    python deploy.py patch     # 0.3.0 -> 0.3.1
    python deploy.py minor     # 0.3.0 -> 0.4.0
    python deploy.py major     # 0.3.0 -> 1.0.0
    python deploy.py --dry-run patch   # Run tests + bump version, skip upload

Requires:
    - pytest, build, twine (pip install pytest build twine)
    - A PyPI API token stored in .pypi_token (one line, no whitespace)
"""

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


def run_tests() -> bool:
    print("\n========== Running tests ==========\n")
    result = subprocess.run([sys.executable, "-m", "pytest", "-v"], cwd=Path(__file__).parent)
    return result.returncode == 0


def clean_dist() -> None:
    if DIST_DIR.exists():
        import shutil
        shutil.rmtree(DIST_DIR)


def build_package() -> bool:
    print("\n========== Building package ==========\n")
    result = subprocess.run([sys.executable, "-m", "build"], cwd=Path(__file__).parent)
    return result.returncode == 0


def upload_to_pypi(token: str) -> bool:
    print("\n========== Uploading to PyPI ==========\n")
    result = subprocess.run(
        [
            sys.executable, "-m", "twine", "upload",
            "--username", "__token__",
            "--password", token,
            "dist/*",
        ],
        cwd=Path(__file__).parent,
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


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    args = [a for a in args if a != "--dry-run"]

    if len(args) != 1 or args[0] not in ("major", "minor", "patch"):
        print(__doc__)
        sys.exit(1)

    bump_type = args[0]
    current = read_current_version()
    new_version = bump_version(current, bump_type)

    print(f"Current version : {current}")
    print(f"Bump type       : {bump_type}")
    print(f"New version     : {new_version}")

    if not dry_run:
        token = load_token()

    if not run_tests():
        sys.exit("\nTests failed — aborting deployment.")

    write_version(new_version)
    print(f"\nVersion updated: {current} -> {new_version}")

    clean_dist()

    if not build_package():
        write_version(current)
        sys.exit("\nBuild failed — version rolled back.")

    if dry_run:
        print("\n[DRY RUN] Skipping PyPI upload.")
        print(f"Build artifacts are in {DIST_DIR}")
        return

    if not upload_to_pypi(token):
        sys.exit("\nUpload to PyPI failed. Built artifacts remain in dist/.")

    print(f"\nSuccessfully deployed phynetpy {new_version} to PyPI!")


if __name__ == "__main__":
    main()
