"""Auth/chat JSON stores and protected API routes (local data under DATA_DIR)."""
from __future__ import annotations

import pytest


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    return tmp_path


def test_auth_store_create_and_verify(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api import auth_store

    uid = auth_store.create_user("student@syr.edu", "secret12", "Ada", "Lovelace")
    assert uid
    assert auth_store.verify_user("student@syr.edu", "secret12") == uid
    assert auth_store.verify_user("student@syr.edu", "wrong") is None


def test_auth_store_duplicate_email(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api import auth_store

    auth_store.create_user("dup@syr.edu", "secret12", "A", "B")
    with pytest.raises(ValueError, match="already registered"):
        auth_store.create_user("dup@syr.edu", "otherpass", "C", "D")


def test_chat_store_thread_and_messages(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api import auth_store, chat_store

    uid = auth_store.create_user("u@syr.edu", "secret12", "Test", "User")
    tid = chat_store.create_thread(uid, "My chat")
    assert chat_store.assert_thread_owned(uid, tid)
    chat_store.append_turn(uid, tid, "hello", "hi there", "general")
    msgs = chat_store.get_messages(uid, tid)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user" and msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "assistant" and msgs[1]["content"] == "hi there"


def test_clear_all_chats(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api import auth_store, chat_store

    uid = auth_store.create_user("clear@syr.edu", "secret12", "C", "User")
    tid = chat_store.create_thread(uid, "T")
    chat_store.append_turn(uid, tid, "a", "b", None)
    assert chat_store.list_threads(uid)
    chat_store.clear_all_chats()
    assert chat_store.list_threads(uid) == []
    assert chat_store.get_messages(uid, tid) == []


def test_signup_login_and_chats_api(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/auth/signup",
            json={
                "email": "api@syr.edu",
                "password": "secret12",
                "first_name": "Api",
                "last_name": "User",
            },
        )
        assert r.status_code == 200
        body = r.json()
        token = body["access_token"]
        assert body["user_id"]
        assert body.get("first_name") == "Api"
        assert body.get("last_name") == "User"

        r2 = client.get("/chats", headers={"Authorization": f"Bearer {token}"})
        assert r2.status_code == 200
        assert r2.json() == []

        r3 = client.post(
            "/chats",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r3.status_code == 200
        thread = r3.json()
        assert thread["id"]

        r4 = client.get(
            f"/chats/{thread['id']}/messages",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r4.status_code == 200
        assert r4.json() == []


def test_chats_require_auth(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.get("/chats")
        assert r.status_code == 401


def test_chat_stream_requires_auth(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/chat/stream",
            json={"message": "hi", "thread_id": "00000000-0000-0000-0000-000000000000"},
        )
        assert r.status_code == 401


def test_chat_post_requires_auth(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/chat",
            json={
                "message": "hi",
                "thread_id": "00000000-0000-0000-0000-000000000000",
            },
        )
        assert r.status_code == 401


def test_chat_post_unknown_thread_returns_404(data_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(data_dir))
    from api.main import app
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        r = client.post(
            "/auth/signup",
            json={
                "email": "t@syr.edu",
                "password": "secret12",
                "first_name": "T",
                "last_name": "User",
            },
        )
        assert r.status_code == 200
        token = r.json()["access_token"]
        r2 = client.post(
            "/chat",
            json={
                "message": "hi",
                "thread_id": "00000000-0000-0000-0000-000000000001",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r2.status_code == 404
