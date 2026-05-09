import pytest
import pandas as pd
import numpy as np
from src.models.predictor import PolyPredictor
from unittest.mock import patch, MagicMock

@pytest.fixture
def predictor():
    return PolyPredictor()

def test_prepare_data(predictor):
    data = {
        "timestamp": [1, 2, 3],
        "datetime": ["a", "b", "c"],
        "token_id": ["t1", "t2", "t3"],
        "market_id": ["m1", "m2", "m3"],
        "question": ["q", "q", "q"],
        "target": [0.5, 0.6, 0.7],
        "feature_1": [0.1, 0.2, 0.3],
        "feature_2": [10, 20, 30]
    }
    df = pd.DataFrame(data)
    
    X, y = predictor._prepare_data(df)
    
    assert list(X.columns) == ["feature_1", "feature_2"]
    assert predictor.feature_names == ["feature_1", "feature_2"]
    np.testing.assert_array_almost_equal(y.values, [0.5, 0.6, 0.7])

def test_predict(predictor):
    predictor.model = MagicMock()
    predictor.model.predict.return_value = np.array([0.55, 0.65])
    predictor.feature_names = ["feature_1"]
    
    data = pd.DataFrame({"feature_1": [0.1, 0.2], "timestamp": [1, 2]})
    preds = predictor.predict(data)
    
    np.testing.assert_array_almost_equal(preds, [0.55, 0.65])
    # Verify predict was called with correct columns
    predictor.model.predict.assert_called_once()
    called_df = predictor.model.predict.call_args[0][0]
    assert list(called_df.columns) == ["feature_1"]
