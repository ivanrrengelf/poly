import os

import numpy as np
import pandas as pd
import pytest

from src.models.predictor import PolyPredictor


def _sample_df(n: int = 40) -> pd.DataFrame:
    timestamps = np.arange(1700000000, 1700000000 + n)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "datetime": pd.to_datetime(timestamps, unit="s"),
            "token_id": ["t1"] * n,
            "market_id": ["m1"] * n,
            "question": ["q"] * n,
            "f1": np.linspace(0.1, 0.9, n),
            "f2": np.linspace(0.9, 0.1, n),
            "target": [i % 2 for i in range(n)],
        }
    )


def test_prepare_data_splits_features_and_target():
    predictor = PolyPredictor()
    df = _sample_df()

    x, y = predictor._prepare_data(df)

    assert y.name == "target"
    assert "target" not in x.columns
    assert set(predictor.feature_names) == {"f1", "f2"}


def test_load_data_missing_file_raises():
    predictor = PolyPredictor()
    with pytest.raises(FileNotFoundError):
        predictor.load_data("does-not-exist.csv")


def test_predict_raises_when_model_missing():
    predictor = PolyPredictor()
    with pytest.raises(ValueError):
        predictor.predict(pd.DataFrame({"f1": [1.0]}))


def test_predict_uses_feature_names_when_present():
    class DummyModel:
        def predict(self, x):
            return np.ones(len(x)) * 0.7

    predictor = PolyPredictor()
    predictor.model = DummyModel()
    predictor.feature_names = ["f1", "f2"]

    out = predictor.predict(pd.DataFrame({"f1": [1.0], "f2": [2.0], "extra": [3.0]}))
    assert np.allclose(out, np.array([0.7]))


def test_save_model_creates_file(tmp_path, monkeypatch):
    class DummyModel:
        def save_model(self, path):
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("dummy")

    predictor = PolyPredictor()
    predictor.model = DummyModel()

    monkeypatch.setattr("src.models.predictor.MODELS_DIR", str(tmp_path))
    predictor.save_model("model.txt")

    assert (tmp_path / "model.txt").exists()


def test_load_model_missing_file_raises(tmp_path, monkeypatch):
    predictor = PolyPredictor()
    monkeypatch.setattr("src.models.predictor.MODELS_DIR", str(tmp_path))

    with pytest.raises(FileNotFoundError):
        predictor.load_model("missing.txt")
