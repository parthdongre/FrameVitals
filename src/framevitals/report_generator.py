from __future__ import annotations

from pathlib import Path

from framevitals.pdf_report_builder import generate_pdf_report as _generate_pdf_report


REPORT_DIR = Path("reports")


def generate_pdf_report(result):
    """Generate a PDF report, creating the output directory only on demand."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    return _generate_pdf_report(
        result,
        output_dir=REPORT_DIR,
        report_title="FrameVitals Dataset Report",
    )
