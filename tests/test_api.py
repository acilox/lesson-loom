"""FastAPI console smoke tests."""

from fastapi.testclient import TestClient

from lesson_loom.app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_generate_endpoint():
    r = client.post("/generate", json={"subject": "science", "content_type": "explainer"})
    assert r.status_code == 200
    assert "body" in r.json()


def test_optimize_and_dashboard(tmp_path, monkeypatch):
    monkeypatch.setenv("LESSON_LOOM_DB", str(tmp_path / "t.sqlite"))
    # reimport-free: app reads DB env at call time via module global is fine for smoke
    r = client.post("/optimize", json={"subject": "science", "rounds": 8})
    assert r.status_code == 200
    body = r.json()
    assert body["best_test_northstar"] > body["baseline_test_northstar"]
    assert client.get("/").status_code == 200
