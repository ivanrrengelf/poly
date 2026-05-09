"""Modelo de predicción para mercados de Polymarket.

Utiliza LightGBM para predecir el precio continuo futuro 
basado en features técnicas y de microestructura.
"""
import os
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error

from config.settings import ModelHyperparams
from src.utils.logger import get_logger

log = get_logger("predictor")

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")


class PolyPredictor:
    """Modelo de predicción de precios usando LightGBM."""

    def __init__(self, hyperparams: ModelHyperparams | None = None):
        self.hp = hyperparams or ModelHyperparams()
        self.model = None
        self.feature_names = []

    def load_data(self, filepath: str | None = None) -> pd.DataFrame:
        """Carga el dataset preprocesado de features."""
        if filepath is None:
            filepath = os.path.join(PROCESSED_DIR, "features.csv")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"No se encontró el dataset en {filepath}")

        df = pd.read_csv(filepath)
        log.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df

    def _prepare_data(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """Separa las features de la variable target."""
        # Filtrar columnas que no son features
        drop_cols = ["timestamp", "datetime", "token_id", "market_id", "question", "target"]
        self.feature_names = [c for c in df.columns if c not in drop_cols]
        
        X = df[self.feature_names]
        y = df["target"]
        return X, y

    def train_walk_forward(self, df: pd.DataFrame):
        """Entrena el modelo usando división temporal estricta (no aleatoria)."""
        log.info("Preparando entrenamiento Walk-Forward...")
        
        # Asegurar orden temporal
        df = df.sort_values("timestamp").reset_index(drop=True)
        
        # Split temporal (ej. 80% pasado para train, 20% futuro para test)
        split_idx = int(len(df) * (1 - self.hp.validation_split))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]
        
        log.info(f"Set de Train: {len(train_df)} filas (hasta {train_df['datetime'].iloc[-1]})")
        log.info(f"Set de Test:  {len(test_df)} filas (desde {test_df['datetime'].iloc[0]})")

        X_train, y_train = self._prepare_data(train_df)
        X_test, y_test = self._prepare_data(test_df)

        # Configurar LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

        params = {
            "objective": "regression",
            "metric": ["rmse", "mae"],
            "boosting_type": "gbdt",
            "learning_rate": self.hp.learning_rate,
            "num_leaves": self.hp.num_leaves,
            "max_depth": self.hp.max_depth,
            "min_child_samples": self.hp.min_child_samples,
            "subsample": self.hp.subsample,
            "colsample_bytree": self.hp.colsample_bytree,
            "reg_alpha": self.hp.reg_alpha,
            "reg_lambda": self.hp.reg_lambda,
            "random_state": self.hp.random_seed,
            "verbosity": -1
        }

        log.info("Entrenando modelo LightGBM...")
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=self.hp.n_estimators,
            valid_sets=[train_data, test_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=self.hp.early_stopping_rounds),
                lgb.log_evaluation(period=50)
            ]
        )
        
        # Evaluar
        self._evaluate(X_test, y_test)
        
        # Guardar modelo
        self.save_model()

    def _evaluate(self, X_test: pd.DataFrame, y_test: pd.Series):
        """Calcula y muestra las métricas en el set de prueba."""
        preds_diff = self.model.predict(X_test)

        mse = mean_squared_error(y_test, preds_diff)
        mae = mean_absolute_error(y_test, preds_diff)

        log.info("=" * 40)
        log.info("RESULTADOS EN TEST (DIFERENCIA DE PRECIO)")
        log.info("=" * 40)
        log.info(f"MSE: {mse:.4f} (El objetivo es < 0.0001)")
        log.info(f"MAE: {mae:.4f} (Un error menor a 0.01 / 1 céntimo es el objetivo para ser competitivos en el spread)")
        
        # Feature Importance
        importance = self.model.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({"feature": self.feature_names, "gain": importance})
        imp_df = imp_df.sort_values(by="gain", ascending=False).head(10)
        
        log.info("\nTOP 10 FEATURES MÁS IMPORTANTES:")
        for _, row in imp_df.iterrows():
            log.info(f"  {row['feature']:<20}: {row['gain']:.2f}")
        log.info("=" * 40)

    def save_model(self, filename: str = "lightgbm_poly.txt"):
        """Guarda el modelo entrenado."""
        os.makedirs(MODELS_DIR, exist_ok=True)
        path = os.path.join(MODELS_DIR, filename)
        self.model.save_model(path)
        log.info(f"Modelo guardado en {path}")

    def load_model(self, filename: str = "lightgbm_poly.txt"):
        """Carga un modelo guardado."""
        path = os.path.join(MODELS_DIR, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"No se encontró el modelo en {path}")
        self.model = lgb.Booster(model_file=path)
        log.info(f"Modelo cargado desde {path}")

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Genera precios predictivos para datos nuevos."""
        if self.model is None:
            raise ValueError("El modelo no está entrenado o cargado.")
        
        # Asegurar que las columnas coinciden
        X = features[self.feature_names] if self.feature_names else features
        return self.model.predict(X)


if __name__ == "__main__":
    predictor = PolyPredictor()
    
    try:
        data = predictor.load_data()
        if not data.empty:
            predictor.train_walk_forward(data)
    except FileNotFoundError as e:
        log.error(e)
