"""Parse government PDFs (IRR, guidebooks) into per-page normalized records."""

import io
import re

import pdfplumber

# Table-of-contents lines end in a run of dot leaders followed by a page number
# (e.g. "Section 7. Membership ....................... 75"). A single wrapped
# ToC entry can span two physical lines, so filtering line-by-line leaves
# truncated fake headers behind. Instead, drop entire pages that are
# ToC-dominated (most lines end in dot leaders, or the page is titled
# "TABLE OF CONTENTS") — real body pages never look like this.
_TOC_LINE_RE = re.compile(r"\.{3,}\s*\d+\s*$")
_TOC_PAGE_RATIO_THRESHOLD = 0.5


def _is_toc_page(lines: list[str]) -> bool:
    content_lines = [line for line in lines if line.strip()]
    if not content_lines:
        return False
    if "TABLE OF CONTENTS" in content_lines[0].upper():
        return True
    dot_leader_count = sum(1 for line in content_lines if _TOC_LINE_RE.search(line))
    return (dot_leader_count / len(content_lines)) > _TOC_PAGE_RATIO_THRESHOLD


def parse_pdf(raw_bytes: bytes, source_url: str) -> list[dict]:
    """Returns a list of {raw_text, page_number, source_url}, one entry per PDF page.

    Pages that are predominantly table-of-contents are dropped entirely.
    """
    pages = []
    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            raw_text = page.extract_text() or ""
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
