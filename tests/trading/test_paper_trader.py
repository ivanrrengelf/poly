import json

import pandas as pd

from config.settings import TradingConfig
from src.trading.paper_trader import PaperTrader


class DummyModel:
    def predict(self, x):
        return [0.7] * len(x)


def _sample_market_df() -> pd.DataFrame:
    ts = list(range(1700000000, 1700000000 + 6))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="s"),
            "market_id": ["m1"] * 6,
            "token_id": ["t1"] * 6,
            "question": ["Will asset move up?"] * 6,
            "price": [0.40, 0.41, 0.42, 0.43, 0.44, 0.45],
            "target": [1, 1, 1, 0, 1, 0],
        }
    )


def test_paper_trader_opens_and_closes_positions(tmp_path):
    config = TradingConfig(
        initial_capital=100.0,
        min_edge_threshold=0.05,
        holding_period_periods=2,
    )
    trader = PaperTrader(config=config, journal_path=str(tmp_path / "journal.json"))
    df = _sample_market_df()
    predictions = DummyModel().predict(df)

    result = trader.run(df, predictions)

    assert result.metrics["closed_trades"] >= 1
    assert result.metrics["open_positions"] == 0
    assert result.chart_data
    assert result.recent_trades

    first_trade = result.trade_journal[0]
    assert first_trade["entry_price"] == 0.4
    assert first_trade["exit_price"] == 0.42
    assert first_trade["holding_period"] == 2
    assert first_trade["entry_time"] != first_trade["exit_time"]


def test_paper_trader_persists_journal(tmp_path):
    config = TradingConfig(initial_capital=100.0, holding_period_periods=2)
    journal_path = tmp_path / "paper_journal.json"
    trader = PaperTrader(config=config, journal_path=str(journal_path))
    df = _sample_market_df()
    predictions = DummyModel().predict(df)

    result = trader.run(df, predictions)

    assert journal_path.exists()
    payload = json.loads(journal_path.read_text(encoding="utf-8"))
    assert payload["metrics"]["closed_trades"] == result.metrics["closed_trades"]
    assert len(payload["trade_journal"]) == len(result.trade_journal)