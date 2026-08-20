"""Shared Qdrant client factory.

Defaults to Qdrant's embedded local mode (no server process needed) so the
pipeline runs without Docker. Set QDRANT_URL (e.g. from docker-compose) to
point at a real Qdrant server instead — same client API either way.

Cached as a process-wide singleton: embedded/local mode holds an exclusive
file lock on qdrant_local/, so two live QdrantClient instances pointed at
the same path — even in the same process — collide with "already accessed
by another instance of Qdrant client" (hit in testing: a bootstrap function
that created one client to check the collection, then called a helper that
created a second one internally, before the first had been garbage
collected). A single shared instance for the process's whole lifetime avoids
that entire class of bug, including under concurrent requests.
"""

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from qdrant_client import QdrantClient

load_dotenv()

LOCAL_STORAGE_PATH = Path(__file__).resolve().parents[2] / "qdrant_local"


@lru_cache(maxsize=1)
def get_client() -> QdrantClient:
    url = os.getenv("QDRANT_URL")
    if url:
        return QdrantClient(url=url)
    LOCAL_STORAGE_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(LOCAL_STORAGE_PATH))
