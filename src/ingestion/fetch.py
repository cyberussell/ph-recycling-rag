"""Download and cache source documents, recording fetch date + checksum.

Government hosts here 403 the default `requests` / `curl` user agent (confirmed
against officialgazette.gov.ph and emb.gov.ph), so every request goes out with a
browser UA. lawphil.net and nswmc.emb.gov.ph resolve fine either way.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

from .source_manifest import SourceDoc, sources_for_milestone

BROWSER_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
MANIFEST_PATH = RAW_DIR / "source_manifest.json"


def _ext_for(fmt: str) -> str:
    return "html" if fmt == "html" else "pdf"


def fetch_doc(doc: SourceDoc, raw_dir: Path = RAW_DIR) -> dict:
    raw_dir.mkdir(parents=True, exist_ok=True)
    resp = requests.get(doc.url, headers={"User-Agent": BROWSER_UA}, timeout=30)
    resp.raise_for_status()

    dest = raw_dir / f"{doc.doc_id}.{_ext_for(doc.fmt)}"
    dest.write_bytes(resp.content)

    checksum = hashlib.sha256(resp.content).hexdigest()
    return {
        "doc_id": doc.doc_id,
        "title": doc.title,
        "url": doc.url,
        "doc_type": doc.doc_type,
        "fmt": doc.fmt,
        "jurisdiction": doc.jurisdiction,
        "language": doc.language,
        "local_path": str(dest.relative_to(raw_dir.parent.parent)),
        "fetch_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "sha256": checksum,
        "bytes": len(resp.content),
    }


def fetch_all(milestone: str = "m1", raw_dir: Path = RAW_DIR) -> list[dict]:
    docs = sources_for_milestone(milestone)
    manifest_entries = []
    for doc in docs:
        print(f"Fetching {doc.doc_id} ({doc.url}) ...")
        entry = fetch_doc(doc, raw_dir)
        manifest_entries.append(entry)
        print(f"  -> {entry['bytes']:,} bytes, sha256={entry['sha256'][:12]}...")

    raw_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "source_manifest.json").write_text(
        json.dumps(manifest_entries, indent=2), encoding="utf-8"
    )
    return manifest_entries


if __name__ == "__main__":
    import sys

    milestone = sys.argv[1] if len(sys.argv) > 1 else "m1"
    fetch_all(milestone)
