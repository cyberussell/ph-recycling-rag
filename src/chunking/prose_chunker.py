"""Chunker for non-statute documents (framework, advisory/guidebook PDFs).

Unlike RA 9003/the IRR, these documents have no numbered Section/Rule
structure to key citations off of — they're narrative guidebooks with
loose heading conventions (ALL-CAPS titles, "A. LETTERED CATEGORIES."). So
citations here are page-based instead of section-based, and chunking is
page-driven with heading detection on a best-effort basis, rather than the
regex hard-boundary approach the legal chunker uses. Output schema matches
section_chunker.py's so the rest of the pipeline is chunker-agnostic.
"""

import re

TARGET_CHUNK_TOKENS = 400


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


_HEADING_RE = re.compile(r"^[A-Z][A-Z0-9 ,\-'/()]{4,}$")
_LETTERED_HEADING_RE = re.compile(r"^[A-Z]\.\s+[A-Z]")


def _detect_heading(lines: list[str]) -> str | None:
    for line in lines[:3]:
        stripped = line.strip()
        if not stripped or len(stripped) > 80:
            continue
        if _LETTERED_HEADING_RE.match(stripped) or (
            _HEADING_RE.match(stripped) and len(stripped.split()) >= 2
        ):
            return stripped
    return None


def chunk_prose_document(
    pages: list[dict], doc_id: str, source_title: str, doc_type: str
) -> list[dict]:
    chunks = []
    for page in pages:
        lines = [l for l in page["raw_text"].split("\n") if l.strip()]
        if not lines:
            continue
        heading = _detect_heading(lines)

        # Split an oversized page into ~TARGET_CHUNK_TOKENS windows at line
        # boundaries so no single chunk balloons past a usable embedding size.
        windows: list[list[str]] = []
        current: list[str] = []
        for line in lines:
            current.append(line)
            if _approx_tokens(" ".join(current)) >= TARGET_CHUNK_TOKENS:
                windows.append(current)
                current = []
        if current:
            windows.append(current)

        for part_idx, window_lines in enumerate(windows):
            text = "\n".join(window_lines).strip()
            if not text:
                continue
            chunk_id = f"{doc_id}-p{page['page_number']}"
            if len(windows) > 1:
                chunk_id += f"-{part_idx + 1}"

            label = f"p.{page['page_number']}"
            header = f"[{source_title}, {label}"
            if heading:
                header += f" — {heading}"
            header += "]"

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source_title": source_title,
                    "doc_type": doc_type,
                    "section_id": None,
                    "section_title": heading,
                    "subsection": None,
                    "page_number": page["page_number"],
                    "hierarchy": "",
                    "text": text,
                    "embedded_text": f"{header}\n{text}",
                }
            )
    return chunks
