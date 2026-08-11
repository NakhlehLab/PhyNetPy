"""Render an NSF report Markdown file to PDF via WeasyPrint.

Usage: python reports/build_pdf.py reports/NSF_Annual_Report_Year1.md
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import markdown

CSS = """
@page {
    size: letter;
    margin: 1in 1in 0.9in 1in;
    @bottom-center {
        content: counter(page) " of " counter(pages);
        font-family: "Times New Roman", serif;
        font-size: 9pt;
        color: #555;
    }
}
body {
    font-family: "Times New Roman", Georgia, serif;
    font-size: 10.5pt;
    line-height: 1.42;
    color: #111;
}
h1 {
    font-size: 17pt;
    margin: 0 0 0.35em 0;
    line-height: 1.25;
}
h2 {
    font-size: 13pt;
    margin: 1.5em 0 0.45em 0;
    padding-bottom: 0.15em;
    border-bottom: 1px solid #bbb;
    page-break-after: avoid;
}
h3 {
    font-size: 11.5pt;
    margin: 1.15em 0 0.35em 0;
    page-break-after: avoid;
}
h4 {
    font-size: 10.5pt;
    font-style: italic;
    margin: 1em 0 0.3em 0;
    page-break-after: avoid;
}
p { margin: 0 0 0.55em 0; text-align: justify; }
ul, ol { margin: 0 0 0.6em 0; padding-left: 1.4em; }
li { margin-bottom: 0.28em; }
hr { border: none; border-top: 1px solid #ccc; margin: 1.3em 0; }
code {
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 0.86em;
    background: #f2f2f2;
    padding: 0.05em 0.28em;
    border-radius: 2px;
}
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6em 0 0.9em 0;
    font-size: 9.5pt;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid #bbb;
    padding: 0.32em 0.5em;
    text-align: left;
    vertical-align: top;
}
th { background: #ececec; font-weight: bold; }
a { color: #12448a; text-decoration: none; }
blockquote {
    margin: 0.6em 0;
    padding-left: 0.9em;
    border-left: 3px solid #ccc;
    color: #444;
}
em { font-style: italic; }
"""


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2

    src = Path(sys.argv[1]).resolve()
    out = src.with_suffix(".pdf")

    body = markdown.markdown(
        src.read_text(encoding="utf-8"),
        extensions=["tables", "sane_lists", "attr_list"],
    )
    html = (
        f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )

    tmp = src.with_suffix(".build.html")
    tmp.write_text(html, encoding="utf-8")
    try:
        subprocess.run(["weasyprint", str(tmp), str(out)], check=True)
    finally:
        tmp.unlink(missing_ok=True)

    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
