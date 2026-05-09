import pandas as pd

from src.features import engineering


def _base_prices_df() -> pd.DataFrame:
    rows = []
    for i in range(12):
        rows.append(
            {
                "market_id": "m1",
                "timestamp": 1700000000 + i * 60,
                "price": 0.4 + i * 0.01,
            }
        )
    for i in range(12):
        rows.append(
            {
                "market_id": "m2",
                "timestamp": 1700000000 + i * 60,
                "price": 0.6 - i * 0.005,
            }
        )
    return pd.DataFrame(rows)


def test_add_lag_features_creates_expected_columns():
    df = _base_prices_df()
    out = engineering.add_lag_features(df.copy())
    assert "price_lag_1" in out.columns
    assert "price_lag_3" in out.columns


def test_add_rolling_features_creates_expected_columns():
    df = _base_prices_df()
    out = engineering.add_rolling_features(df.copy())
    assert "rolling_mean_5" in out.columns
    assert "rolling_std_10" in out.columns
    assert "rolling_min_30" in out.columns
    assert "rolling_max_60" in out.columns


def test_add_momentum_features_creates_expected_columns():
    df = _base_prices_df()
    out = engineering.add_momentum_features(df.copy())
    assert "momentum_1" in out.columns
    assert "roc_10" in out.columns
    assert "acceleration" in out.columns


def test_add_volatility_features_creates_expected_columns():
    df = _base_prices_df()
    out = engineering.add_volatility_features(df.copy())
    assert "volatility_10" in out.columns
    assert "volatility_30" in out.columns
    assert "range_pct_10" in out.columns
    assert "range_pct_30" in out.columns


def test_add_position_features_value_range():
    df = _base_prices_df()
    out = engineering.add_position_features(df.copy())
    assert "position_30" in out.columns
    assert "position_60" in out.columns
    assert (out["position_30"].dropna().between(0, 1)).all()


def test_add_trade_features_with_empty_trades_sets_defaults():
    prices = _base_prices_df()
    trades = pd.DataFrame()
    out = engineering.add_trade_features(prices.copy(), trades)
    assert (out["trade_count_1h"] == 0).all()
    assert (out["trade_volume_1h"] == 0.0).all()
    assert (out["buy_pressure_1h"] == 0.5).all()


def test_add_trade_features_uses_only_past_trades():
    prices = pd.DataFrame(
        [
            {"market_id": "m1", "timestamp": 1000, "price": 0.5},
            {"market_id": "m1", "timestamp": 2000, "price": 0.6},
        ]
    )
    trades = pd.DataFrame(
        [
            {"market_id": "m1", "timestamp": 900, "size": 10, "side": "BUY"},
            {"market_id": "m1", "timestamp": 1500, "size": 4, "side": "SELL"},
            {"market_id": "m1", "timestamp": 2100, "size": 99, "side": "BUY"},
        ]
    )
    out = engineering.add_trade_features(prices.copy(), trades)
    assert out.loc[0, "trade_count_1h"] == 1
    assert out.loc[1, "trade_count_1h"] == 2


def test_add_target_creates_binary_column():
    df = _base_prices_df()
    out = engineering.add_target(df.copy(), horizon=2)
    assert "target" in out.columns
    assert set(out["target"].dropna().unique()).issubset({0, 1})
