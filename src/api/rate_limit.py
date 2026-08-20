"""In-memory request throttle for the public demo deployment.

Deliberately simple: no Redis, no external store. A public portfolio demo
running on a single free-tier container doesn't need a distributed limiter —
it needs a cheap hard cap so a burst of traffic (or abuse) can't run up an
unbounded Anthropic API bill. State resets on container restart, which is
fine for this use case.

Two caps, both configurable via env vars:
- per-IP daily cap (stops one visitor from hammering it)
- global daily cap (hard ceiling on total cost regardless of traffic shape)
"""

import os
import threading
from collections import defaultdict
from datetime import date

PER_IP_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_PER_IP_PER_DAY", "8"))
GLOBAL_DAILY_LIMIT = int(os.getenv("RATE_LIMIT_GLOBAL_PER_DAY", "80"))

_lock = threading.Lock()
_per_ip_counts: dict[str, int] = defaultdict(int)
_global_count = 0
_current_day = date.today()


def _roll_day_if_needed() -> None:
    global _current_day, _global_count, _per_ip_counts
    today = date.today()
    if today != _current_day:
        _current_day = today
        _global_count = 0
        _per_ip_counts = defaultdict(int)


def check_and_increment(client_ip: str) -> tuple[bool, str]:
    """Returns (allowed, reason_if_blocked)."""
    global _global_count
    with _lock:
        _roll_day_if_needed()

        if _global_count >= GLOBAL_DAILY_LIMIT:
            return False, "Daily demo request limit reached — please try again tomorrow."
        if _per_ip_counts[client_ip] >= PER_IP_DAILY_LIMIT:
            return False, "You've hit today's per-visitor limit for this demo — please try again tomorrow."

        _global_count += 1
        _per_ip_counts[client_ip] += 1
        return True, ""
