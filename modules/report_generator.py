from __future__ import annotations

from pathlib import Path

from modules.pdf_report_builder import generate_pdf_report as _generate_pdf_report

REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)


def generate_pdf_report(result):
    return _generate_pdf_report(result, output_dir=REPORT_DIR)
