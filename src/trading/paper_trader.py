"""Paper trading engine for fixed-horizon, cash-based simulations."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import pandas as pd

from config.settings import TradingConfig


@dataclass(frozen=True)
class ClosedTrade:
    trade_id: int
    market_id: str
    question: str
    token_id: str
    entry_time: str
    exit_time: str
    entry_price: float
    exit_price: float
    shares: float
    bet_size: float
    open_fee: float
    close_fee: float
    pnl: float
    return_pct: float
    holding_period: int
    probability: float
    edge: float
    status: str


@dataclass(frozen=True)
class PaperTradingResult:
    metrics: dict[str, Any]
    chart_data: list[dict[str, Any]]
    recent_trades: list[dict[str, Any]]
    trade_journal: list[dict[str, Any]]
    open_positions: list[dict[str, Any]]
    skipped_signals: int


class PaperTrader:
    """Executes a long-only paper trading simulation with fixed exits."""

    def __init__(self, config: TradingConfig | None = None, journal_path: str | None = None):
        self.config = config or TradingConfig()
        resolved_journal_path = journal_path or os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "models", "paper_trading_journal.json"
        )
        self.journal_path = os.path.abspath(resolved_journal_path)

    def run(self, df: pd.DataFrame, predictions: Sequence[float]) -> PaperTradingResult:
        """Run a paper trading backtest using a fixed holding period."""
        if len(df) != len(predictions):
            raise ValueError(
                "The number of predictions must match the number of rows in the dataset."
            )

        if df.empty:
            raise ValueError("The trading dataset cannot be empty.")

        market_df = df.sort_values("timestamp").reset_index(drop=True).copy()
        probabilities = [float(probability) for probability in predictions]

        cash = self.config.initial_capital
        open_positions: list[dict[str, Any]] = []
        closed_trades: list[ClosedTrade] = []
        chart_data: list[dict[str, Any]] = []
        trade_id = 1
        skipped_signals = 0

        for index, row in market_df.iterrows():
            current_time = self._serialize_time(row["datetime"])
            current_price = float(row["price"])

            if current_price < self.config.min_entry_price or current_price > self.config.max_entry_price:
                cash, open_positions, closed_now = self._close_due_positions(
                    open_positions=open_positions,
                    current_index=index,
                    current_time=current_time,
                    current_price=current_price,
                    cash=cash,
                )
                closed_trades.extend(closed_now)
                equity = cash + sum(pos["shares"] * current_price for pos in open_positions)
                chart_data.append(
                    {
                        "time": current_time,
                        "value": equity,
                        "cash": cash,
                        "open_positions": len(open_positions),
                    }
                )
                continue

            cash, open_positions, closed_now = self._close_due_positions(
                open_positions=open_positions,
                current_index=index,
                current_time=current_time,
                current_price=current_price,
                cash=cash,
            )
            closed_trades.extend(closed_now)

            probability = probabilities[index]
            edge = probability - current_price

            if edge > self.config.min_edge_threshold and len(open_positions) < self.config.max_concurrent_positions:
                bet_size = min(
                    cash * self.config.max_position_pct * self.config.kelly_fraction,
                    self.config.max_trade_size_eur,
                )
                open_fee = bet_size * self.config.execution_fee_pct
                total_debit = bet_size + open_fee

                if bet_size > 0 and cash >= total_debit:
                    position = self._open_position(
                        trade_id=trade_id,
                        row=row,
                        current_index=index,
                        probability=probability,
                        edge=edge,
                        bet_size=bet_size,
                        open_fee=open_fee,
                    )
                    cash -= total_debit
                    open_positions.append(position)
                    trade_id += 1
                else:
                    skipped_signals += 1
            elif edge > self.config.min_edge_threshold:
                skipped_signals += 1

            equity = cash + sum(pos["shares"] * current_price for pos in open_positions)
            chart_data.append(
                {
                    "time": current_time,
                    "value": equity,
                    "cash": cash,
                    "open_positions": len(open_positions),
                }
            )

        final_time = self._serialize_time(market_df.iloc[-1]["datetime"])
        final_price = float(market_df.iloc[-1]["price"])
        cash, open_positions, forced_closes = self._close_remaining_positions(
            open_positions=open_positions,
            final_time=final_time,
            final_price=final_price,
            cash=cash,
        )
        closed_trades.extend(forced_closes)

        if forced_closes:
            chart_data.append(
                {
                    "time": final_time,
                    "value": cash,
                    "cash": cash,
                    "open_positions": len(open_positions),
                }
            )

        total_closed = len(closed_trades)
        wins = sum(1 for trade in closed_trades if trade.pnl >= 0)
        realized_pnl = cash - self.config.initial_capital
        win_rate = (wins / total_closed) * 100 if total_closed else 0.0

        trade_journal = [asdict(trade) for trade in closed_trades]
        result = PaperTradingResult(
            metrics={
                "initial_capital": self.config.initial_capital,
                "final_capital": cash,
                "realized_pnl": realized_pnl,
                "roi_pct": (realized_pnl / self.config.initial_capital) * 100,
                "closed_trades": total_closed,
                "open_positions": len(open_positions),
                "win_rate": win_rate,
                "skipped_signals": skipped_signals,
            },
            chart_data=chart_data,
            recent_trades=trade_journal[-50:],
            trade_journal=trade_journal,
            open_positions=[self._serialize_position(position) for position in open_positions],
            skipped_signals=skipped_signals,
        )
        self._persist_journal(result)
        return result

    def _open_position(
        self,
        trade_id: int,
        row: pd.Series,
        current_index: int,
        probability: float,
        edge: float,
        bet_size: float,
        open_fee: float,
    ) -> dict[str, Any]:
        entry_price = float(row["price"])
        effective_entry_price = entry_price * (1 + self.config.slippage_pct)
        holding_period = self.config.holding_period_periods

        return {
            "trade_id": trade_id,
            "market_id": str(row.get("market_id", "UNKNOWN")),
            "question": self._market_label(row),
            "token_id": str(row.get("token_id", "UNKNOWN")),
            "entry_time": self._serialize_time(row["datetime"]),
            "entry_price": entry_price,
            "effective_entry_price": effective_entry_price,
            "shares": bet_size / effective_entry_price,
            "bet_size": bet_size,
            "open_fee": open_fee,
            "probability": probability,
            "edge": edge,
            "entry_index": current_index,
            "close_index": current_index + holding_period,
            "holding_period": holding_period,
        }

    def _close_due_positions(
        self,
        open_positions: list[dict[str, Any]],
        current_index: int,
        current_time: str,
        current_price: float,
        cash: float,
    ) -> tuple[float, list[dict[str, Any]], list[ClosedTrade]]:
        remaining_positions: list[dict[str, Any]] = []
        closed_positions: list[ClosedTrade] = []

        for position in open_positions:
            if position["close_index"] <= current_index:
                cash, closed_trade = self._close_position(
                    position=position,
                    exit_time=current_time,
                    exit_price=current_price,
                    cash=cash,
                    final_close=False,
                )
                closed_positions.append(closed_trade)
            else:
                remaining_positions.append(position)

        return cash, remaining_positions, closed_positions

    def _close_remaining_positions(
        self,
        open_positions: list[dict[str, Any]],
        final_time: str,
        final_price: float,
        cash: float,
    ) -> tuple[float, list[dict[str, Any]], list[ClosedTrade]]:
        closed_positions: list[ClosedTrade] = []

        while open_positions:
            position = open_positions.pop(0)
            cash, closed_trade = self._close_position(
                position=position,
                exit_time=final_time,
                exit_price=final_price,
                cash=cash,
                final_close=True,
            )
            closed_positions.append(closed_trade)

        return cash, open_positions, closed_positions

    def _close_position(
        self,
        position: dict[str, Any],
        exit_time: str,
        exit_price: float,
        cash: float,
        final_close: bool,
    ) -> tuple[float, ClosedTrade]:
        effective_exit_price = exit_price * (1 - self.config.slippage_pct)
        gross_proceeds = position["shares"] * effective_exit_price
        close_fee = gross_proceeds * self.config.execution_fee_pct
        cash += gross_proceeds - close_fee

        pnl = gross_proceeds - close_fee - position["bet_size"] - position["open_fee"]
        return_pct = (pnl / (position["bet_size"] + position["open_fee"])) * 100
        status = "WIN" if pnl >= 0 else "LOSS"

        trade = ClosedTrade(
            trade_id=position["trade_id"],
            market_id=position["market_id"],
            question=position["question"],
            token_id=position["token_id"],
            entry_time=position["entry_time"],
            exit_time=exit_time,
            entry_price=position["entry_price"],
            exit_price=exit_price,
            shares=position["shares"],
            bet_size=position["bet_size"],
            open_fee=position["open_fee"],
            close_fee=close_fee,
            pnl=pnl,
            return_pct=return_pct,
            holding_period=position["holding_period"],
            probability=position["probability"],
            edge=position["edge"],
            status=status if not final_close else f"{status}_FINAL",
        )
        return cash, trade

    def _persist_journal(self, result: PaperTradingResult) -> None:
        os.makedirs(os.path.dirname(self.journal_path), exist_ok=True)
        payload = {
            "metrics": result.metrics,
            "chart_data": result.chart_data,
            "recent_trades": result.recent_trades,
            "trade_journal": result.trade_journal,
            "open_positions": result.open_positions,
        }
        with open(self.journal_path, "w", encoding="utf-8") as file_handle:
            json.dump(payload, file_handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _serialize_time(value: Any) -> str:
        return pd.to_datetime(value).to_pydatetime().isoformat()

    @staticmethod
    def _market_label(row: pd.Series) -> str:
        question = row.get("question")
        if question is None or (isinstance(question, float) and pd.isna(question)):
            return f"Market {row.get('market_id', 'UNKNOWN')}"
        return str(question)

    @staticmethod
    def _serialize_position(position: dict[str, Any]) -> dict[str, Any]:
        return {
            "trade_id": position["trade_id"],
            "market_id": position["market_id"],
            "question": position["question"],
            "token_id": position["token_id"],
            "entry_time": position["entry_time"],
            "entry_price": position["entry_price"],
            "bet_size": position["bet_size"],
            "probability": position["probability"],
            "edge": position["edge"],
            "close_index": position["close_index"],
        }
