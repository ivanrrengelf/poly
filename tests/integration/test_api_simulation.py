import pandas as pd
from fastapi.testclient import TestClient

from src.dashboard import api as dashboard_api


class DummyModel:
    def predict(self, x):
        return [0.7] * len(x)


class DummyPredictor:
    def __init__(self):
        self.hp = type("HP", (), {"validation_split": 0.2})()
        self.feature_names = ["f1"]
        self.model = DummyModel()

    def load_data(self):
        n = 10
        ts = list(range(1700000000, 1700000000 + n))
        return pd.DataFrame(
            {
                "timestamp": ts,
                "datetime": pd.to_datetime(ts, unit="s"),
                "token_id": ["tok-1"] * n,
                "market_id": ["m1"] * n,
                "question": ["q"] * n,
                "price": [0.4] * n,
                "f1": [0.1] * n,
                "target": [1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
            }
        )

    def train_walk_forward(self, df):
        return None


def test_run_simulation_missing_data(monkeypatch):
    dashboard_api._SIMULATION_CACHE = None
    monkeypatch.setattr(dashboard_api.os.path, "exists", lambda p: False)

    result = dashboard_api.run_simulation()

    assert result["status"] == "MISSING_DATA"
    assert "error" in result


def test_run_simulation_success(monkeypatch):
    dashboard_api._SIMULATION_CACHE = None
    monkeypatch.setattr(dashboard_api.os.path, "exists", lambda p: True)
    monkeypatch.setattr(dashboard_api, "PolyPredictor", DummyPredictor)

    result = dashboard_api.run_simulation()

    assert "metrics" in result
    assert "chart_data" in result
    assert "recent_trades" in result
    assert "runtime_seconds" in result["metrics"]
    assert "entry_time" in result["recent_trades"][0]
    assert "exit_time" in result["recent_trades"][0]


def test_api_endpoint_returns_json(monkeypatch):
    dashboard_api._SIMULATION_CACHE = None
    monkeypatch.setattr(dashboard_api.os.path, "exists", lambda p: True)
    monkeypatch.setattr(dashboard_api, "PolyPredictor", DummyPredictor)

    client = TestClient(dashboard_api.app)
    response = client.get("/api/simulation")

    assert response.status_code == 200
    body = response.json()
    assert "metrics" in body
    assert "trade_journal" in body


def test_api_health_endpoint():
    client = TestClient(dashboard_api.app)
    response = client.get("/api/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "timestamp" in body


def test_root_endpoint():
    client = TestClient(dashboard_api.app)
    response = client.get("/")

    assert response.status_code == 200
    assert "message" in response.json()
