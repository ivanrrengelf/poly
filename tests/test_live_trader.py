import pytest
import sqlite3
import datetime
import os
import asyncio
import uuid
from unittest.mock import patch, MagicMock

# Set test DB path BEFORE importing LiveTrader
import src.dashboard.live_trader as lt

@pytest.fixture
def test_trader():
    test_db_path = f"test_portfolio_{uuid.uuid4().hex}.db"
    lt.DB_PATH = test_db_path
    
    # Mock TradingConfig to allow modifying attributes
    class MockConfig:
        min_edge_threshold = 0.05
        kelly_fraction = 0.25
        max_position_pct = 0.1
        min_liquidity_usd = 1000.0
        min_volume_24h_usd = 500.0
        initial_capital = 100.0
        take_profit_pct = 0.15
        stop_loss_pct = -0.2
        
    lt.tc = MockConfig()
    
    trader = lt.LiveTrader()
    # Mocking external API clients
    trader.clob = MagicMock()
    trader.gamma = MagicMock()
    
    yield trader
    
    # Ensure connections are closed by garbage collection before deleting
    import gc
    gc.collect()
    if os.path.exists(test_db_path):
        try:
            os.remove(test_db_path)
        except PermissionError:
            pass # On Windows sometimes it takes a moment to release the lock

@pytest.mark.asyncio
async def test_manage_open_trades_roi_calculation(test_trader):
    # Setup initial capital (total: 1000, 200 locked in 2 active trades)
    test_trader.update_portfolio_state(1000.0, 800.0)
    
    # Mock current price via clob
    async def mock_get_book(market_id):
        if market_id == "m_long_win":
            return {"bids": [{"price": "0.6"}], "asks": [{"price": "0.6"}]}
        elif market_id == "m_short_win":
            return {"bids": [{"price": "0.3"}], "asks": [{"price": "0.3"}]}
        return {"bids": [{"price": "0.5"}], "asks": [{"price": "0.5"}]}
        
    test_trader.clob.get_book = mock_get_book
    
    now = datetime.datetime.now()
    target_close = now + datetime.timedelta(hours=1)
    
    with sqlite3.connect(lt.DB_PATH) as conn:
        cursor = conn.cursor()
        # LONG trade: entered at 0.5, current price is 0.6
        # Expected ROI: (0.6 - 0.5)/0.5 = 0.2 - 0.01 (fee) = 0.19
        cursor.execute('''
            INSERT INTO active_trades 
            (market_id, question, type, entry_price, predicted_prob, edge, bet_size, liquidity_at_entry, entry_time, target_close_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ("m_long_win", "Q_Long", "LONG", 0.5, 0.7, 0.1, 100.0, 1000.0, now.isoformat(), target_close.isoformat()))
        
        # SHORT trade: entered at 0.5, current price is 0.3
        # Expected ROI: (0.5 - 0.3)/(1 - 0.5) = 0.2/0.5 = 0.4 - 0.01 = 0.39
        cursor.execute('''
            INSERT INTO active_trades 
            (market_id, question, type, entry_price, predicted_prob, edge, bet_size, liquidity_at_entry, entry_time, target_close_time)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ("m_short_win", "Q_Short", "SHORT", 0.5, 0.3, 0.1, 100.0, 1000.0, now.isoformat(), target_close.isoformat()))
        conn.commit()
    
    # Set take_profit very low to force close
    lt.tc.take_profit_pct = 0.15
    lt.tc.stop_loss_pct = -0.5
    
    await test_trader.manage_open_trades()
    
    # Verify both trades were closed due to take profit
    with sqlite3.connect(lt.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM active_trades')
        assert cursor.fetchone()[0] == 0
        
        cursor.execute('SELECT pnl, roi FROM historical_trades WHERE market_id="m_long_win"')
        long_trade = cursor.fetchone()
        assert long_trade is not None
        assert abs(long_trade[1] - 0.19) < 0.001
        assert abs(long_trade[0] - 19.0) < 0.001 # 100 * 0.19
        
        cursor.execute('SELECT pnl, roi FROM historical_trades WHERE market_id="m_short_win"')
        short_trade = cursor.fetchone()
        assert short_trade is not None
        assert abs(short_trade[1] - 0.39) < 0.001
        assert abs(short_trade[0] - 39.0) < 0.001 # 100 * 0.39
        
    total_cap, avail_cap = test_trader.get_portfolio_state()
    # Initial 1000 - 200 (invested) = 800 available.
    # Returns: 100 + 19 = 119 for long. 100 + 39 = 139 for short.
    # Total available = 800 + 119 + 139 = 1058. Total capital = 1058.
    assert abs(total_cap - 1058.0) < 0.001
    assert abs(avail_cap - 1058.0) < 0.001

@pytest.mark.asyncio
async def test_execute_live_predictions_edge_calculation(test_trader):
    import pandas as pd
    import numpy as np
    
    test_trader.update_portfolio_state(1000.0, 1000.0)
    
    # Mock data
    mock_df = pd.DataFrame({
        "market_id": ["m_buy", "m_sell"],
        "question": ["Q1", "Q2"],
        "liquidity": [5000.0, 5000.0]
    })
    
    test_trader.fetch_live_data = MagicMock(return_value=asyncio.sleep(0, mock_df))
    test_trader.calculate_live_features = MagicMock(return_value=mock_df.copy())
    
    # Mock predictor output
    test_trader.predictor.feature_names = []
    test_trader.predictor.model = MagicMock()
    # model predicts expected_price of 0.8 for m_buy and 0.2 for m_sell
    test_trader.predictor.model.predict.return_value = np.array([0.8, 0.2])
    
    async def mock_get_book(market_id):
        if market_id == "m_buy":
            # Book has ask at 0.6 -> Expected 0.8 -> Edge long = 0.2
            return {"bids": [{"price": "0.58"}], "asks": [{"price": "0.6"}]}
        elif market_id == "m_sell":
            # Book has bid at 0.4 -> Expected 0.2 -> Edge short = 0.2
            return {"bids": [{"price": "0.4"}], "asks": [{"price": "0.42"}]}
        return {"bids": [{"price": "0.5"}], "asks": [{"price": "0.5"}]}
    
    test_trader.clob.get_book = mock_get_book
    lt.tc.min_edge_threshold = 0.1
    lt.tc.max_position_pct = 0.1
    lt.tc.kelly_fraction = 1.0
    
    # Wait for execution
    await test_trader.execute_live_predictions()
    
    # Verify active trades
    with sqlite3.connect(lt.DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('SELECT market_id, type, entry_price, edge, bet_size FROM active_trades')
        trades = cursor.fetchall()
        
    assert len(trades) == 2
    
    # Verify m_buy (LONG)
    long_trade = next(t for t in trades if t[0] == "m_buy")
    assert long_trade[1] == "LONG"
    # best_bid + 0.01 = 0.59
    assert abs(long_trade[2] - 0.59) < 0.001
    assert abs(long_trade[3] - 0.2) < 0.001 # Edge = 0.8 - 0.6 = 0.2
    
    # Verify m_sell (SHORT)
    short_trade = next(t for t in trades if t[0] == "m_sell")
    assert short_trade[1] == "SHORT"
    # best_ask - 0.01 = 0.41
    assert abs(short_trade[2] - 0.41) < 0.001
    assert abs(short_trade[3] - 0.2) < 0.001 # Edge = 0.4 - 0.2 = 0.2
