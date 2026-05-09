import pytest
import pandas as pd
import numpy as np
from src.features.engineering import add_target, add_lag_features, add_momentum_features

def test_add_target():
    # Test that add_target correctly calculates the continuous expected price
    data = {
        "market_id": ["m1", "m1", "m1", "m1", "m2", "m2", "m2"],
        "price": [0.1, 0.2, 0.3, 0.4, 0.9, 0.8, 0.7]
    }
    df = pd.DataFrame(data)
    
    # Using horizon 2 for m1 and m2
    df_result = add_target(df, horizon=2)
    
    # For m1: 0.1 -> 0.3, 0.2 -> 0.4, 0.3 -> NaN, 0.4 -> NaN
    # For m2: 0.9 -> 0.7, 0.8 -> NaN, 0.7 -> NaN
    expected_targets = [0.3, 0.4, np.nan, np.nan, 0.7, np.nan, np.nan]
    
    np.testing.assert_array_almost_equal(df_result["target"].values, expected_targets)

def test_add_lag_features():
    # Setup mock config to avoid reading global settings that might depend on files
    from src.features import engineering
    original_cfg = engineering.cfg
    class MockConfig:
        lag_periods = [1, 2]
    engineering.cfg = MockConfig()
    
    data = {
        "market_id": ["m1", "m1", "m1", "m2", "m2"],
        "price": [0.1, 0.2, 0.3, 0.5, 0.6]
    }
    df = pd.DataFrame(data)
    
    try:
        df_result = add_lag_features(df)
        assert "price_lag_1" in df_result.columns
        assert "price_lag_2" in df_result.columns
        
        # For m1: price_lag_1 -> [NaN, 0.1, 0.2]
        np.testing.assert_array_almost_equal(
            df_result[df_result["market_id"] == "m1"]["price_lag_1"].values,
            [np.nan, 0.1, 0.2]
        )
        
        # For m2: price_lag_2 -> [NaN, NaN]
        np.testing.assert_array_almost_equal(
            df_result[df_result["market_id"] == "m2"]["price_lag_2"].values,
            [np.nan, np.nan]
        )
    finally:
        engineering.cfg = original_cfg

def test_add_momentum_features():
    data = {
        "market_id": ["m1", "m1", "m1", "m1"],
        "price": [0.1, 0.15, 0.12, 0.20]
    }
    df = pd.DataFrame(data)
    
    df_result = add_momentum_features(df)
    
    assert "momentum_1" in df_result.columns
    # diff(1) -> [NaN, 0.05, -0.03, 0.08]
    np.testing.assert_array_almost_equal(
        df_result["momentum_1"].values,
        [np.nan, 0.05, -0.03, 0.08]
    )
