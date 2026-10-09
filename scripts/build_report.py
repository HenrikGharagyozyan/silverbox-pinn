"""Build report.pdf from report.md: Markdown -> HTML -> PDF with headless Chrome."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report.md"
OUT = ROOT / "report.pdf"

CSS = """
@page { size: A4; margin: 18mm 16mm; }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.45; color: #111; }
h1 { font-size: 18pt; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; padding-bottom: 2pt; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; }
th, td { border: 1px solid #bbb; padding: 2pt 6pt; text-align: left; }
th { background: #f0f0f0; }
img { max-width: 100%; display: block; margin: 8pt auto 2pt; }
p:has(> img) { break-inside: avoid; break-after: avoid; }
blockquote { margin: 6pt 0 6pt 16pt; font-size: 11.5pt; }
code { font-size: 9pt; }
em { color: #333; }
"""

BROWSERS = ["google-chrome", "chromium", "chromium-browser", "google-chrome-stable"]


def main() -> None:
    browser = next((b for b in BROWSERS if shutil.which(b)), None)
    if browser is None:
        sys.exit("no Chrome/Chromium found; cannot print the PDF")

    body = markdown.markdown(SRC.read_text(encoding="utf-8"), extensions=["tables"])
    html = (
        f'<!doctype html><html><head><meta charset="utf-8"><base href="{ROOT.as_uri()}/">'
        f"<style>{CSS}</style></head><body>{body}</body></html>"
    )
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run(
            [
                browser, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                "--allow-file-access-from-files", f"--print-to-pdf={OUT}", page.as_uri(),
            ],
            check=True,
            capture_output=True,
        )
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
