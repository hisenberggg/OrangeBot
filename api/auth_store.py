"""Email/password users persisted in JSON (development only).

Passwords are stored in plaintext for simplicity—do not use in production.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from api.app_data import ensure_data_dir

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _users_path():
    return ensure_data_dir() / "users.json"


def _load() -> dict[str, Any]:
    path = _users_path()
    if not path.exists():
        return {"users": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict) or "users" not in data:
            return {"users": []}
        return data
    except (json.JSONDecodeError, OSError):
        return {"users": []}


def _save(data: dict[str, Any]) -> None:
    path = _users_path()
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def validate_email(email: str) -> bool:
    return bool(email and _EMAIL_RE.match(email.strip()))


def create_user(
    email: str,
    password: str,
    first_name: str,
    last_name: str,
) -> str:
    """Register user; returns user_id. Raises ValueError on duplicate or invalid input."""
    email = email.strip().lower()
    fn = first_name.strip()
    ln = last_name.strip()
    if not validate_email(email):
        raise ValueError("Invalid email")
    if len(password) < 6:
        raise ValueError("Password must be at least 6 characters")
    if not fn or not ln:
        raise ValueError("First and last name are required")

    data = _load()
    users: list[dict[str, Any]] = data["users"]
    for u in users:
        if u.get("email") == email:
            raise ValueError("Email already registered")

    user_id = str(uuid.uuid4())
    users.append(
        {
            "user_id": user_id,
            "email": email,
            "password": password,
            "first_name": fn,
            "last_name": ln,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    _save(data)
    return user_id


def verify_user(email: str, password: str) -> str | None:
    """Return user_id if credentials match, else None."""
    email = email.strip().lower()
    data = _load()
    for u in data["users"]:
        if u.get("email") == email:
            stored = u.get("password")
            if stored is None:
                return None
            return str(u["user_id"]) if stored == password else None
    return None


def get_user_by_id(user_id: str) -> dict[str, Any] | None:
    data = _load()
    for u in data["users"]:
        if u.get("user_id") == user_id:
            return u
    return None
