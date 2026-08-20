"""Section-aware chunker for Philippine legal/regulatory text.

Different document types use different structural conventions:
  - RA 9003 (lawphil.net HTML): "CHAPTER I" / "Article 1" / "Section 1.\n<title>\n- <body>"
  - IRR of RA 9003 (NSWMC PDF):  "PART I." / "Rule I. <title>" / "Section 1. <title>\n<body>"

Rather than force one regex to fit both, each doc_id gets a small structural
config. The output chunk schema is identical regardless of source format, so
everything downstream (embedding, indexing, retrieval) is format-agnostic.
"""

import re
from dataclasses import dataclass, field

MAX_CHUNK_TOKENS = 700
TARGET_CHUNK_TOKENS = 450


def _approx_tokens(text: str) -> int:
    """Cheap word-count-based proxy; good enough for chunk-size decisions."""
    return int(len(text.split()) * 1.3)


@dataclass
class DocStructureConfig:
    doc_id: str
    source_title: str  # short name used in citation headers, e.g. "RA 9003"
    doc_type: str
    top_level_re: re.Pattern  # chapter/part
    top_level_label: str  # "Chapter" | "Part"
    mid_level_re: re.Pattern | None  # article/rule (may be None)
    mid_level_label: str
    section_re: re.Pattern
    section_title_on_next_line: bool
    subsection_re: re.Pattern = field(
        default_factory=lambda: re.compile(r"^\(([a-zA-Z0-9]{1,3})\)\s+")
    )


CONFIGS: dict[str, DocStructureConfig] = {
    "ra9003": DocStructureConfig(
        doc_id="ra9003",
        source_title="RA 9003",
        doc_type="statute",
        top_level_re=re.compile(r"^CHAPTER\s*([IVXLCDM]+)\s*$"),
        top_level_label="Chapter",
        mid_level_re=re.compile(r"^Article\s*(\d+)\s*$"),
        mid_level_label="Article",
        section_re=re.compile(r"^Section\s*(\d+)\.\s*$"),
        section_title_on_next_line=True,
    ),
    "ra9003-irr": DocStructureConfig(
        doc_id="ra9003-irr",
        source_title="IRR of RA 9003",
        doc_type="irr",
        top_level_re=re.compile(r"^PART\s+([IVXLCDM]+)\.?\s*(.*)$"),
        top_level_label="Part",
        mid_level_re=re.compile(r"^Rule\s+([IVXLCDM]+)\.\s*(.*)$"),
        mid_level_label="Rule",
        section_re=re.compile(r"^Section\s+(\d+)\.\s*(.*)$"),
        section_title_on_next_line=False,
    ),
}


def _build_offset_page_map(pages: list[dict]) -> tuple[str, list[tuple[int, int]]]:
    """Concatenate page texts into one stream; return (full_text, [(offset, page_number), ...])."""
    parts = []
    breakpoints = []
    offset = 0
    for p in pages:
        breakpoints.append((offset, p["page_number"]))
        parts.append(p["raw_text"])
        offset += len(p["raw_text"]) + 1  # +1 for the join newline
    return "\n".join(parts), breakpoints


def _page_for_offset(offset: int, breakpoints: list[tuple[int, int]]) -> int:
    page = breakpoints[0][1]
    for bp_offset, bp_page in breakpoints:
        if bp_offset <= offset:
            page = bp_page
        else:
            break
    return page


def _split_by_subsections(body: str, subsection_re: re.Pattern) -> list[tuple[str | None, str]]:
    """Split section body into [(subsection_label_or_None, text), ...] at lettered/numbered markers."""
    lines = body.split("\n")
    groups: list[tuple[str | None, list[str]]] = []
    current_label = None
    current_lines: list[str] = []

    for line in lines:
        m = subsection_re.match(line)
        if m:
            if current_lines:
                groups.append((current_label, current_lines))
            current_label = m.group(1)
            current_lines = [line]
        else:
            current_lines.append(line)
    if current_lines:
        groups.append((current_label, current_lines))

    return [(label, "\n".join(ls).strip()) for label, ls in groups if "\n".join(ls).strip()]


def chunk_document(pages: list[dict], config: DocStructureConfig) -> list[dict]:
    """Returns a list of chunk dicts ready for embedding/indexing (schema per plan's Qdrant payload)."""
    full_text, breakpoints = _build_offset_page_map(pages)
    lines = full_text.split("\n")

    # Pass 1: walk lines, tracking hierarchy + section boundaries with char offsets.
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1

    current_top = None
    current_mid = None
    sections: list[dict] = []  # {section_id, section_title, top, mid, start_line, start_offset}

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()

        top_m = config.top_level_re.match(line)
        if top_m:
            current_top = f"{config.top_level_label} {top_m.group(1)}"
            i += 1
            continue

        if config.mid_level_re:
            mid_m = config.mid_level_re.match(line)
            if mid_m:
                current_mid = f"{config.mid_level_label} {mid_m.group(1)}"
                i += 1
                continue

        sec_m = config.section_re.match(line)
        if sec_m:
            sec_num = sec_m.group(1)
            if config.section_title_on_next_line:
                sec_title = lines[i + 1].strip() if i + 1 < n else ""
                body_start_line = i + 2
            else:
                sec_title = sec_m.group(2).strip() if sec_m.lastindex and sec_m.lastindex >= 2 else ""
                body_start_line = i + 1

            sections.append(
                {
                    "section_id": f"Sec. {sec_num}",
                    "section_title": sec_title,
                    "top": current_top,
                    "mid": current_mid,
                    "body_start_line": body_start_line,
                    "header_start_offset": line_offsets[i],
                }
            )
            i = body_start_line
            continue

        i += 1

    # Pass 2: body of each section = text from its body_start_line to the next section's header line.
    chunks: list[dict] = []
    for idx, sec in enumerate(sections):
        body_start = sec["body_start_line"]
        next_header_offset = (
            sections[idx + 1]["header_start_offset"] if idx + 1 < len(sections) else len(full_text)
        )
        body_start_offset = line_offsets[body_start] if body_start < len(line_offsets) else len(full_text)
        body_text = full_text[body_start_offset:next_header_offset].strip()
        # Drop a leading "- " (RA 9003 HTML convention) since it's a formatting artifact, not content.
        body_text = re.sub(r"^-\s*", "", body_text)

        page_number = _page_for_offset(sec["header_start_offset"], breakpoints)

        base_meta = {
            "doc_id": config.doc_id,
            "source_title": config.source_title,
            "doc_type": config.doc_type,
            "section_id": sec["section_id"],
            "section_title": sec["section_title"],
            "page_number": page_number,
            "hierarchy": " / ".join(x for x in [sec["top"], sec["mid"]] if x),
        }

        if _approx_tokens(body_text) <= MAX_CHUNK_TOKENS:
            chunks.append(_make_chunk(base_meta, subsection=None, text=body_text))
        else:
            sub_groups = _split_by_subsections(body_text, config.subsection_re)
            if len(sub_groups) <= 1:
                # No clean subsection markers to split on; fall back to one oversized chunk
                # rather than silently truncating legal content.
                chunks.append(_make_chunk(base_meta, subsection=None, text=body_text))
            else:
                for label, sub_text in sub_groups:
                    chunks.append(_make_chunk(base_meta, subsection=label, text=sub_text))

    return _dedupe_chunk_ids(chunks)


def _dedupe_chunk_ids(chunks: list[dict]) -> list[dict]:
    """Some source documents (e.g. the IRR) restart section numbering within a
    single Rule across unnumbered sub-bodies, producing structurally-identical
    ids for genuinely different provisions. Guarantee point-id uniqueness for
    Qdrant so a later chunk never silently overwrites an earlier one."""
    seen: dict[str, int] = {}
    for chunk in chunks:
        cid = chunk["chunk_id"]
        seen[cid] = seen.get(cid, 0) + 1
        if seen[cid] > 1:
            chunk["chunk_id"] = f"{cid}-dup{seen[cid]}"
    return chunks


def _make_chunk(base_meta: dict, subsection: str | None, text: str) -> dict:
    sec_label = base_meta["section_id"] + (f"({subsection})" if subsection else "")
    header_prefix = f"{base_meta['hierarchy']}, " if base_meta["hierarchy"] else ""
    header = f"[{base_meta['source_title']}, {header_prefix}{sec_label}"
    if base_meta["section_title"]:
        header += f" — {base_meta['section_title']}"
    header += "]"

    # Section numbers restart within each Rule/Article in these documents, so the
    # hierarchy slug (e.g. "ruleIII") must be part of the id or cross-rule chunks collide.
    hierarchy_slug = re.sub(r"[^a-zA-Z0-9]+", "", base_meta["hierarchy"])
    chunk_id = f"{base_meta['doc_id']}-{hierarchy_slug}-{base_meta['section_id'].replace('Sec. ', 'sec')}"
    if subsection:
        chunk_id += f"-{subsection}"

    return {
        "chunk_id": chunk_id,
        "doc_id": base_meta["doc_id"],
        "source_title": base_meta["source_title"],
        "doc_type": base_meta["doc_type"],
        "section_id": base_meta["section_id"],
        "section_title": base_meta["section_title"],
        "subsection": subsection,
        "page_number": base_meta["page_number"],
        "hierarchy": base_meta["hierarchy"],
        "text": text,
        "embedded_text": f"{header}\n{text}",
    }
