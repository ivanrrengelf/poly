"""Backend FastAPI para el Dashboard de Simulación de Polymarket.

Ejecuta una simulación de paper trading (Backtest) usando el modelo predictivo
y sirve los resultados a la interfaz web.
"""
import os
import time
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from config.settings import TradingConfig
from src.models.predictor import PolyPredictor
from src.trading.paper_trader import PaperTrader
from src.utils.logger import get_logger

log = get_logger("dashboard_api")

app = FastAPI(title="Polymarket Prediction Dashboard")
FEATURES_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "processed", "features.csv"
)
REQUIRED_FEATURE_COLUMNS = {
    "timestamp",
    "price",
    "datetime",
    "token_id",
    "question",
    "target",
}

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rutas de archivos
PUBLIC_DIR = os.path.join(os.path.dirname(__file__), "public")
os.makedirs(PUBLIC_DIR, exist_ok=True)

# Montar frontend estático
app.mount("/app", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")

# Variables globales para cachear la simulación
_SIMULATION_CACHE = None
tc = TradingConfig()


def _validate_features_schema(df: pd.DataFrame) -> None:
    """Valida que el CSV de features tenga las columnas mínimas esperadas."""
    missing_columns = sorted(REQUIRED_FEATURE_COLUMNS.difference(df.columns))
    if missing_columns:
        raise ValueError(
            "El dataset de features no cumple el esquema requerido. "
            f"Faltan columnas: {', '.join(missing_columns)}"
        )

    if df.empty:
        raise ValueError("El dataset de features está vacío")


def _load_and_validate_features(predictor: PolyPredictor) -> pd.DataFrame:
    """Carga el CSV de features y normaliza el esquema mínimo esperado."""
    df = predictor.load_data()
    _validate_features_schema(df)

    df = df.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    if df["datetime"].isna().any():
        raise ValueError("La columna datetime contiene valores inválidos")

    if (df["price"] <= 0).any():
        raise ValueError("La columna price contiene valores no positivos")

    if (df["price"] > 1).any():
        raise ValueError("La columna price contiene valores fuera del rango [0, 1]")

    return df.sort_values("timestamp").reset_index(drop=True)


def run_simulation():
    """Ejecuta el backtest en los datos de prueba y cachea los resultados."""
    global _SIMULATION_CACHE
    if _SIMULATION_CACHE is not None:
        return _SIMULATION_CACHE

    log.info("Iniciando simulación de paper trading...")

    if not os.path.exists(FEATURES_PATH):
        error_msg = (
            f"Dataset de features no encontrado en {FEATURES_PATH}. "
            "Ejecuta primero: collector.py → engineering.py"
        )
        log.error(error_msg)
        return {"error": error_msg, "status": "MISSING_DATA"}

    predictor = PolyPredictor()
    
    try:
        load_start = time.perf_counter()
        df = _load_and_validate_features(predictor)
        load_elapsed = time.perf_counter() - load_start
        log.info(
            "Datos cargados y validados en %.3fs (%s filas, %s columnas)",
            load_elapsed,
            len(df),
            len(df.columns),
        )

        train_start = time.perf_counter()
        predictor.train_walk_forward(df)
        split_idx = int(len(df) * (1 - predictor.hp.validation_split))
        test_df = df.iloc[split_idx:].copy()
        train_elapsed = time.perf_counter() - train_start
        log.info("Entrenamiento walk-forward completado en %.3fs", train_elapsed)

        predict_start = time.perf_counter()
        X_test = test_df[predictor.feature_names]
        preds_prob = predictor.model.predict(X_test)
        test_df["pred_prob"] = preds_prob

        predict_elapsed = time.perf_counter() - predict_start
        log.info("Predicciones generadas en %.3fs", predict_elapsed)

        simulation_start = time.perf_counter()
        trader = PaperTrader(config=tc)
        trading_result = trader.run(test_df, preds_prob)

        simulation_elapsed = time.perf_counter() - simulation_start
        log.info("Simulación de trades completada en %.3fs", simulation_elapsed)

        portfolio_history = trading_result.chart_data
        trades_history = trading_result.recent_trades
        portfolio_chart = (
            portfolio_history[:: max(1, len(portfolio_history) // 100)]
            if len(portfolio_history) > 100
            else portfolio_history
        )

        _SIMULATION_CACHE = {
            "metrics": {
                **trading_result.metrics,
                "total_trades": trading_result.metrics["closed_trades"],
                "win_rate": trading_result.metrics["win_rate"],
                "runtime_seconds": {
                    "load": round(load_elapsed, 4),
                    "train": round(train_elapsed, 4),
                    "predict": round(predict_elapsed, 4),
                    "simulation": round(simulation_elapsed, 4),
                },
            },
            "chart_data": portfolio_chart,
            "recent_trades": trades_history[-50:],
            "trade_journal": trading_result.trade_journal,
            "open_positions": trading_result.open_positions,
            "journal_path": trader.journal_path,
        }
        
        log.info(
            "Simulación terminada. ROI: %.2f%%",
            _SIMULATION_CACHE["metrics"]["roi_pct"],
        )
        return _SIMULATION_CACHE

    except FileNotFoundError as e:
        error_msg = f"Archivo no encontrado: {e}"
        log.error(error_msg)
        return {"error": error_msg, "status": "FILE_ERROR"}
    except ValueError as e:
        error_msg = f"Error en datos o configuración: {e}"
        log.error(error_msg)
        return {"error": error_msg, "status": "DATA_ERROR"}
    except Exception as e:
        error_msg = f"Error inesperado en simulación: {type(e).__name__}: {e}"
        log.error(error_msg)
        return {"error": error_msg, "status": "RUNTIME_ERROR"}


@app.get("/api/simulation")
async def get_simulation():
    """Endpoint que devuelve los resultados del backtest."""
    return run_simulation()


@app.get("/")
async def root():
    """Redirige al dashboard."""
    return {"message": "API Running. Go to /app/index.html to view the dashboard."}


@app.get("/api/health")
async def healthcheck():
    """Endpoint simple de health para monitoreo."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "simulation_cached": _SIMULATION_CACHE is not None,
        "features_available": os.path.exists(FEATURES_PATH),
    }


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
