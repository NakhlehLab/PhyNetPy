"""
Flag the writing patterns the manuscript is meant to avoid.

Two groups. Inflated adjectives and stock academic transitions are rejected
outright. The second group -- constructions that are acceptable occasionally but
become a tic when repeated -- is counted and reported, so a reviewer can judge
density rather than being told a single use is wrong.

Run: python paper/phynetpy-1.0.0/check_prose.py
Exits non-zero if anything in the first group appears.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path(__file__).resolve().parent

# Inflated language. Rejected outright.
BANNED_WORDS = [
    "groundbreaking", "revolutionary", "transformative", "cutting-edge",
    "powerful", "elegant", "seamless", "seamlessly", "comprehensive",
    "sophisticated", "unprecedented", "exciting", "state-of-the-art",
    "leverage", "leverages", "leveraging", "empowers", "robustly",
    "effortlessly", "vast", "myriad", "plethora", "paradigm",
]

# Stock transitions and filler openers. Rejected outright.
BANNED_PHRASES = [
    "in this work, we", "importantly,", "notably,", "furthermore,",
    "moreover,", "this represents", "taken together", "it is worth noting",
    "at its core", "opens the door", "paves the way", "a wide variety of",
    "a rich set of", "delve into", "it should be noted that",
    "plays a crucial role", "plays a vital role", "we are thrilled",
    "first and foremost", "last but not least", "needless to say",
]

# Acceptable in moderation; reported with counts.
WATCH_PATTERNS = {
    "This provides / This allows / This enables": r"\bThis (?:provides|allows|enables|ensures)\b",
    "not only X but also Y": r"\bnot only\b[^.]{0,80}\bbut also\b",
    "However, at sentence start": r"(?:^|(?<=\. ))However,",
    "In addition, at sentence start": r"(?:^|(?<=\. ))In addition,",
    "crucial / vital / essential": r"\b(?:crucial|vital|essential)\b",
    "significantly / substantially": r"\b(?:significantly|substantially)\b",
    "simply / just": r"\b(?:simply|just)\b",
}


def strip_latex(text: str) -> str:
    """Remove comments, code listings and verbatim so only prose is checked."""
    text = re.sub(r"(?<!\\)%.*", "", text)
    text = re.sub(
        r"\\begin\{lstlisting\}.*?\\end\{lstlisting\}", "", text, flags=re.S
    )
    text = re.sub(r"\\code\{[^}]*\}", " CODE ", text)
    text = re.sub(r"\\texttt\{[^}]*\}", " CODE ", text)
    return text


def main() -> int:
    files = sorted((PAPER / "sections").glob("*.tex"))
    if not files:
        print("no section files found", file=sys.stderr)
        return 1

    hard_hits: list[str] = []
    watch_counts: dict[str, list[str]] = {k: [] for k in WATCH_PATTERNS}

    for path in files:
        raw = path.read_text(encoding="utf-8")
        prose = strip_latex(raw)
        lines = prose.splitlines()

        for i, line in enumerate(lines, 1):
            low = line.lower()
            for word in BANNED_WORDS:
                if re.search(rf"\b{re.escape(word)}\b", low):
                    hard_hits.append(f"{path.name}:{i}: '{word}' -- {line.strip()[:90]}")
            for phrase in BANNED_PHRASES:
                if phrase in low:
                    hard_hits.append(
                        f"{path.name}:{i}: '{phrase}' -- {line.strip()[:90]}"
                    )

        for label, pattern in WATCH_PATTERNS.items():
            for match in re.finditer(pattern, prose):
                line_no = prose[: match.start()].count("\n") + 1
                watch_counts[label].append(f"{path.name}:{line_no}")

    print(f"Checked {len(files)} section files.\n")

    if hard_hits:
        print(f"BANNED ({len(hard_hits)}):")
        for hit in hard_hits:
            print(f"  {hit}")
    else:
        print("BANNED: none found.")

    print("\nWATCH (acceptable in moderation, shown with counts):")
    for label, hits in watch_counts.items():
        if hits:
            print(f"  {len(hits):>3}  {label}")
            print(f"       {', '.join(hits[:8])}"
                  + (" ..." if len(hits) > 8 else ""))

    return 1 if hard_hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
