"""Admin access and paid-call quota tests; no network requests are made."""

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from petsafe.api import admin_store


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(admin_store, "DB_PATH", tmp_path / "admin.sqlite3")
    return admin_store


def test_disabled_calls_and_persistence(store):
    store.save_settings(False, 5)
    with pytest.raises(store.MetaUnavailable, match="api_disabled"):
        store.reserve_call("spark")
    assert store.stats()["spark"]["total"] == 0
    assert store.stats()["enabled"] is False
    # Reopening through the same module reads the on-disk settings.
    store.save_settings(True, 2)
    store.reserve_call("sam")
    assert store.stats()["sam"]["total"] == 1


def test_quota_shared_between_models(store):
    store.save_settings(True, 2)
    store.reserve_call("spark")
    store.reserve_call("sam")
    with pytest.raises(store.MetaUnavailable, match="quota_exceeded"):
        store.reserve_call("spark")
    assert store.stats()["remaining"] == 0


def test_concurrent_reservations_are_atomic(store):
    store.save_settings(True, 4)

    def try_call(_index):
        try:
            store.reserve_call("spark")
        except store.MetaUnavailable as exc:
            return exc.code
        return "reserved"

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(try_call, range(12)))
    assert outcomes.count("reserved") == 4
    assert outcomes.count("quota_exceeded") == 8
    assert store.stats()["spark"]["total"] == 4


def test_rolling_window_and_cost(store, monkeypatch):
    monkeypatch.setattr(admin_store.time, "time", lambda: 100_000.0)
    store.save_settings(True, 1)
    call_id = store.reserve_call("spark")
    store.set_call_cost(call_id, 0.125)
    store.record_investigation()
    assert store.stats()["estimated_cost_usd"]["last_24h"] == 0.125
    monkeypatch.setattr(admin_store.time, "time", lambda: 200_000.0)
    assert store.stats()["remaining"] == 1
    assert store.stats()["spark"] == {"total": 1, "last_24h": 0}
    assert store.stats()["investigations"] == {"total": 1, "last_24h": 0}


def test_admin_routes_require_signed_cookie(store, monkeypatch):
    from fastapi.testclient import TestClient

    from petsafe.api.app import app

    monkeypatch.setenv("PETSAFE_ADMIN_PASSWORD", "test-admin-password")
    monkeypatch.setenv("PETSAFE_SESSION_SECRET", "test-secret-" * 4)
    client = TestClient(app)
    assert client.get("/admin/api/stats").status_code == 401
    assert client.post("/admin/api/settings", json={"enabled": False, "daily_limit": 0}).status_code == 401
    assert client.post("/admin/api/login", json={"password": "wrong"}).status_code == 401
    login = client.post("/admin/api/login", json={"password": "test-admin-password"})
    assert login.status_code == 200
    assert "httponly" in login.headers["set-cookie"].lower()
    assert client.post("/admin/api/settings", json={"enabled": False, "daily_limit": 3}).status_code == 200
    assert client.get("/admin/api/stats").json()["enabled"] is False
    assert client.post("/admin/api/logout").status_code == 200
    assert client.get("/admin/api/stats").status_code == 401


def test_spark_retry_reserves_another_slot_before_network(store, monkeypatch):
    from petsafe.api import services

    class ConnectionError(Exception):
        pass

    class StatusError(Exception):
        pass

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(
        APIConnectionError=ConnectionError, APIStatusError=StatusError,
    ))
    store.save_settings(True, 1)
    attempts = []

    async def failed_create(**_kwargs):
        attempts.append(1)
        raise ConnectionError("simulated transport failure")

    client = SimpleNamespace(chat=SimpleNamespace(
        completions=SimpleNamespace(create=failed_create),
    ))
    with pytest.raises(store.MetaUnavailable, match="quota_exceeded"):
        asyncio.run(services._paid_spark_call(client, model="test"))
    assert len(attempts) == 1
    assert store.stats()["spark"]["total"] == 1
