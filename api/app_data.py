"""Resolved paths for local JSON/SQLite data (gitignored `data/`)."""
from __future__ import annotations

import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent


def get_data_dir() -> Path:
    """Directory for users.json, chats.db, checkpoints.db. Override with DATA_DIR env."""
    raw = os.environ.get("DATA_DIR", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (_ROOT / "data").resolve()


def ensure_data_dir() -> Path:
    d = get_data_dir()
    d.mkdir(parents=True, exist_ok=True)
    return d
