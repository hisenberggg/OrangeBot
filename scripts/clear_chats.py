"""
Remove all chat threads and messages from chats.db.

Uses the same DATA_DIR as the API (env DATA_DIR, default ./data from project root).

Run from project root:  python -m scripts.clear_chats
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    from api.chat_store import clear_all_chats

    clear_all_chats()
    print("Removed all threads and messages from chats.db (under DATA_DIR).")


if __name__ == "__main__":
    main()
