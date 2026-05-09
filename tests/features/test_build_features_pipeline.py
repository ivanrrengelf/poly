import pandas as pd

from src.features import engineering


def test_build_features_end_to_end(tmp_path, monkeypatch):
    markets = pd.DataFrame(
        [{"market_id": "m1", "volume": 100, "volume_24h": 10, "liquidity": 5}]
    )
    prices = pd.DataFrame(
        [
            {
                "market_id": "m1",
                "timestamp": 1700000000 + i * 60,
                "price": 0.4 + i * 0.01,
                "datetime": pd.Timestamp("2023-01-01") + pd.Timedelta(minutes=i),
                "token_id": "t1",
                "question": "q1",
            }
            for i in range(20)
        ]
    )
    trades = pd.DataFrame(
        [
            {"market_id": "m1", "timestamp": 1700000000 + i * 60, "size": 1 + i, "side": "BUY"}
            for i in range(10)
        ]
    )

    monkeypatch.setattr(engineering, "load_raw_data", lambda: (markets, prices, trades))
    monkeypatch.setattr(engineering, "PROCESSED_DIR", str(tmp_path))

    out = engineering.build_features()

    assert not out.empty
    assert "target" in out.columns
    assert (tmp_path / "features.csv").exists()


def test_build_features_returns_empty_when_no_prices(monkeypatch):
    monkeypatch.setattr(
        engineering,
        "load_raw_data",
        lambda: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame()),
    )

    out = engineering.build_features()

    assert out.empty
