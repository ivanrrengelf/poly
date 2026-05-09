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


def _simulate_trade(
    row: pd.Series,
    probability: float,
    capital: float,
) -> tuple[float, dict[str, object] | None, str | None]:
    """Simula una posición YES con precio de entrada/salida y costos."""
    entry_price = float(row["price"])
    exit_price = 1.0 if int(row["target"]) == 1 else 0.0

    if entry_price <= 0:
        return capital, None, "invalid_entry_price"

    bet_size = capital * tc.max_position_pct * tc.kelly_fraction
    if bet_size <= 0:
        return capital, None, "invalid_bet_size"

    effective_entry_price = entry_price * (1 + tc.slippage_pct)
    effective_exit_price = exit_price * (1 - tc.slippage_pct)
    shares = bet_size / effective_entry_price
    gross_proceeds = shares * effective_exit_price
    fees = (bet_size + gross_proceeds) * tc.execution_fee_pct
    pnl = gross_proceeds - bet_size - fees
    new_capital = capital + pnl

    question = row.get("question")
    if question is None or (isinstance(question, float) and pd.isna(question)):
        market_label = f"Market {row.get('market_id', 'UNKNOWN')}"
    else:
        market_label = str(question)

    trade = {
        "time": row["datetime"],
        "market": market_label,
        "prob": float(probability),
        "entry_price": entry_price,
        "exit_price": exit_price,
        "bet_size": float(bet_size),
        "fees": float(fees),
        "pnl": float(pnl),
        "status": "WIN" if pnl >= 0 else "LOSS",
    }
    return new_capital, trade, None


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
        capital = tc.initial_capital
        portfolio_history = []
        trades_history = []

        wins = 0
        losses = 0
        
        for _, row in test_df.iterrows():
            portfolio_history.append({
                "time": row["datetime"],
                "value": capital
            })
            
            prob = row["pred_prob"]
            edge = prob - float(row["price"])

            if edge > tc.min_edge_threshold:
                capital, trade, error = _simulate_trade(row, prob, capital)
                if error is not None:
                    log.warning("Trade omitido en %s: %s", row.get("datetime"), error)
                    continue

                if trade is not None:
                    trades_history.append(trade)
                    if trade["pnl"] >= 0:
                        wins += 1
                    else:
                        losses += 1

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

        simulation_elapsed = time.perf_counter() - simulation_start
        log.info("Simulación de trades completada en %.3fs", simulation_elapsed)
        
        if len(portfolio_history) > 100:
            step = len(portfolio_history) // 100
            portfolio_chart = portfolio_history[::step]
        else:
            portfolio_chart = portfolio_history

        _SIMULATION_CACHE = {
            "metrics": {
                "initial_capital": tc.initial_capital,
                "final_capital": capital,
                "roi_pct": ((capital - tc.initial_capital) / tc.initial_capital) * 100,
                "total_trades": wins + losses,
                "win_rate": win_rate * 100,
                "runtime_seconds": {
                    "load": round(load_elapsed, 4),
                    "train": round(train_elapsed, 4),
                    "predict": round(predict_elapsed, 4),
                    "simulation": round(simulation_elapsed, 4),
                },
            },
            "chart_data": portfolio_chart,
            "recent_trades": trades_history[-50:]
        }
        
        log.info(f"Simulación terminada. ROI: {_SIMULATION_CACHE['metrics']['roi_pct']:.2f}%")
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
