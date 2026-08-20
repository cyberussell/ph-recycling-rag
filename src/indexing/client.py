"""Shared Qdrant client factory.

Defaults to Qdrant's embedded local mode (no server process needed) so the
pipeline runs without Docker. Set QDRANT_URL (e.g. from docker-compose) to
point at a real Qdrant server instead — same client API either way.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

LOCAL_STORAGE_PATH = Path(__file__).resolve().parents[2] / "qdrant_local"


def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    LOCAL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(LOCAL_STORAGE_PATH))
