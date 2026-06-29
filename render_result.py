"""Render outputs/result.md to outputs/result.html and open it in the browser."""
import os
import subprocess
import sys

import markdown

OUT = os.path.join(os.path.dirname(__file__), "outputs")
md_path = os.path.join(OUT, "result.md")
html_path = os.path.join(OUT, "result.html")

with open(md_path, encoding="utf-8") as f:
    body = markdown.markdown(f.read(), extensions=["tables", "fenced_code"])

CSS = """
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 1100px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; color: #222; }
table { border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }
th, td { border: 1px solid #ccc; padding: 6px 10px; vertical-align: top; text-align: left; }
th { background: #f4f4f4; }
tr:nth-child(even) td { background: #fafafa; }
img { max-width: 100%; }
p { margin: 0.35rem 0; }
"""

with open(html_path, "w", encoding="utf-8") as f:
    f.write(f"<!doctype html><meta charset='utf-8'><title>OCR result</title><style>{CSS}</style>{body}")

print(f"wrote {html_path}")
if "--no-open" not in sys.argv:
    subprocess.run(["open", html_path], check=False)
