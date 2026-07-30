"""
Report generation: CSV export (for spreadsheets/further analysis) and
PDF export (for a shareable, printable inspection summary).
"""

import os
import csv
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from src.utils.config import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)


def export_csv(inspections: list, filename: str = None) -> str:
    """Write inspection records to a CSV file and return the file path."""
    if not inspections:
        raise ValueError("No inspection records to export")

    if filename is None:
        filename = f"inspection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    filepath = os.path.join(REPORTS_DIR, filename)
    fieldnames = list(inspections[0].keys())

    with open(filepath, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(inspections)

    logger.info(f"CSV report saved to {filepath}")
    return filepath


def export_pdf(summary: dict, filename: str = None) -> str:
    """
    Generate a simple one-page PDF summary report using reportlab.
    For a factory setting, this is the kind of doc that gets printed and
    pinned to a shift-report board.
    """
    if filename is None:
        filename = f"inspection_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    filepath = os.path.join(REPORTS_DIR, filename)
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height - 60, "AI Factory Vision - Inspection Report")

    c.setFont("Helvetica", 11)
    c.drawString(50, height - 90, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    c.setFont("Helvetica", 13)
    y = height - 140
    lines = [
        f"Total Products Inspected: {summary.get('total', 0)}",
        f"Good Products: {summary.get('good', 0)}",
        f"Defective Products: {summary.get('defective', 0)}",
        f"Defect Rate: {summary.get('defect_rate', 0)}%",
    ]
    for line in lines:
        c.drawString(50, y, line)
        y -= 25

    c.save()
    logger.info(f"PDF report saved to {filepath}")
    return filepath
