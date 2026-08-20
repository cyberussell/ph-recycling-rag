"""Parse government PDFs (IRR, guidebooks) into per-page normalized records."""

import io
import re

import pdfplumber

# Table-of-contents lines contain a run of dot leaders (plain "." or the "…"
# glyph some PDFs use instead), usually but not always ending in a page number
# — long entries can wrap the trailing number onto its own line, so the
# trailing-digit requirement isn't reliable. A run of 3+ dot-like chars
# anywhere in a line essentially never occurs in real prose, so its presence
# alone is enough signal. A single ToC section can also span multiple physical
# pages with no repeated title, so page-level ratio (not just a title check
# on page 1 of the ToC) is what actually catches continuation pages.
_TOC_LINE_RE = re.compile(r"[.…]{3,}")
_TOC_PAGE_RATIO_THRESHOLD = 0.35
_TOC_TITLE_MARKERS = ("TABLE OF CONTENTS", "LIST OF TABLES", "LIST OF FIGURES")


def _is_toc_page(lines: list[str]) -> bool:
    content_lines = [line for line in lines if line.strip()]
    if not content_lines:
        return False
    if any(marker in content_lines[0].upper() for marker in _TOC_TITLE_MARKERS):
        return True
    dot_leader_count = sum(1 for line in content_lines if _TOC_LINE_RE.search(line))
    return (dot_leader_count / len(content_lines)) > _TOC_PAGE_RATIO_THRESHOLD


def _extract_page_text(page) -> str:
    """Landscape pages in this corpus (e.g. the household segregation guide) are
    laid out in two columns; pdfplumber's default extraction reads straight
    across both columns line-by-line and interleaves unrelated sentences. A
    landscape orientation is a reliable signal here, so those pages are
    extracted as left/right column crops and concatenated in reading order.
    Portrait pages (the statute, IRR, framework, other guides) are single-column
    and extract cleanly with the default whole-page call.
    """
    if page.width > page.height:
        midpoint = page.width / 2
        left = page.crop((0, 0, midpoint, page.height)).extract_text() or ""
        right = page.crop((midpoint, 0, page.width, page.height)).extract_text() or ""
        return f"{left}\n{right}"
    return page.extract_text() or ""


def parse_pdf(raw_bytes: bytes, source_url: str) -> list[dict]:
    """Returns a list of {raw_text, page_number, source_url}, one entry per PDF page.

    Pages that are predominantly table-of-contents are dropped entirely.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = _extract_page_text(page)
            lines = raw_text.split("\n")
            if _is_toc_page(lines):
                continue
            text = raw_text.strip()
            if not text:
                continue
            pages.append(
                {"raw_text": text, "page_number": i + 1, "source_url": source_url}
            )
    return pages
