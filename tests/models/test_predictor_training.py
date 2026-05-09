import numpy as np
import pandas as pd

from src.models.predictor import PolyPredictor


class DummyBooster:
    def predict(self, x):
        return np.array([0.6] * len(x))

    def feature_importance(self, importance_type="gain"):
        return np.array([1.0, 2.0])

    def save_model(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("ok")


def _training_df(n: int = 40) -> pd.DataFrame:
    ts = np.arange(1700000000, 1700000000 + n)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "datetime": pd.to_datetime(ts, unit="s"),
            "token_id": ["t"] * n,
            "market_id": ["m"] * n,
            "question": ["q"] * n,
            "f1": np.linspace(0.1, 1.0, n),
            "f2": np.linspace(1.0, 0.1, n),
            "target": [i % 2 for i in range(n)],
        }
    )


def test_train_walk_forward_sets_model(monkeypatch, tmp_path):
    predictor = PolyPredictor()
    df = _training_df()

    class DummyDataset:
        def __init__(self, x, label=None, reference=None):
            self.x = x
            self.label = label
            self.reference = reference

    def fake_train(params, train_data, num_boost_round, valid_sets, callbacks):
        assert params["objective"] == "binary"
        assert num_boost_round == predictor.hp.n_estimators
        assert len(valid_sets) == 2
        return DummyBooster()

    monkeypatch.setattr("src.models.predictor.lgb.Dataset", DummyDataset)
    monkeypatch.setattr("src.models.predictor.lgb.train", fake_train)
    monkeypatch.setattr("src.models.predictor.lgb.early_stopping", lambda stopping_rounds: object())
    monkeypatch.setattr("src.models.predictor.lgb.log_evaluation", lambda period=50: object())
    monkeypatch.setattr("src.models.predictor.MODELS_DIR", str(tmp_path))

    predictor.train_walk_forward(df)

    assert predictor.model is not None
    assert (tmp_path / "lightgbm_poly.txt").exists()


def test_evaluate_runs_with_dummy_model():
    predictor = PolyPredictor()
    predictor.feature_names = ["f1", "f2"]
    predictor.model = DummyBooster()

    x_test = pd.DataFrame({"f1": [0.1, 0.2], "f2": [0.9, 0.8]})
    y_test = pd.Series([1, 0])

    predictor._evaluate(x_test, y_test)
