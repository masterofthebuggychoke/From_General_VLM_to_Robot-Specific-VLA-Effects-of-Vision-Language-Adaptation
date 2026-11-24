"""Render the independent A4 English manuscript and sync the downloadable website PDF."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser", default=shutil.which("google-chrome") or shutil.which("chromium"))
    parser.add_argument("--no-sandbox", action="store_true", help="Use only when the local environment requires it.")
    args = parser.parse_args()
    if not args.browser:
        parser.error("Install Chrome or Chromium")
    root = Path(__file__).resolve().parents[1]
    manuscript = root / "report/manuscript.html"
    destinations = [root / "report/SmolVLA-study.pdf", root / "site/report.pdf"]
    with tempfile.TemporaryDirectory(prefix="smolvla-report-") as work:
        pdf = Path(work) / "rendered.pdf"
        command = [args.browser, "--headless", "--disable-extensions", "--disable-gpu",
                   "--disable-background-networking", "--allow-file-access-from-files",
                   "--no-pdf-header-footer", f"--user-data-dir={Path(work) / 'profile'}",
                   f"--print-to-pdf={pdf}"]
        if args.no_sandbox:
            command.append("--no-sandbox")
        subprocess.run([*command, manuscript.as_uri()], check=True)
        if not pdf.is_file() or not pdf.read_bytes().startswith(b"%PDF-"):
            raise RuntimeError("Missing PDF output")
        for destination in destinations:
            temp = destination.with_suffix(".pdf.tmp")
            shutil.copyfile(pdf, temp)
            temp.replace(destination)
        print(destinations[0])


if __name__ == "__main__":
    main()
