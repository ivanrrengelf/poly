"""Backend FastAPI para el Dashboard de Simulación de Polymarket.

Ejecuta una simulación de paper trading (Backtest) usando el modelo predictivo
y sirve los resultados a la interfaz web.
"""
import os
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


def run_simulation():
    """Ejecuta el backtest en los datos de prueba y cachea los resultados."""
    global _SIMULATION_CACHE
    if _SIMULATION_CACHE is not None:
        return _SIMULATION_CACHE

    log.info("Iniciando simulación de paper trading...")
    
    # Validación de precondiciones: features.csv debe existir
    features_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "processed", "features.csv"
    )
    if not os.path.exists(features_path):
        error_msg = (
            f"Dataset de features no encontrado en {features_path}. "
            "Ejecuta primero: collector.py → engineering.py"
        )
        log.error(error_msg)
        return {"error": error_msg, "status": "MISSING_DATA"}
    
    predictor = PolyPredictor()
    
    try:
        # 1. Cargar datos y modelo
        df = predictor.load_data()
        
        # Simplemente re-entrenamos el walk-forward aquí para asegurar 
        # que tenemos el modelo fresco, pero en un entorno real se cargaría.
        predictor.train_walk_forward(df)
        
        # Obtener los datos de test (último 20%)
        df = df.sort_values("timestamp").reset_index(drop=True)
        split_idx = int(len(df) * (1 - predictor.hp.validation_split))
        test_df = df.iloc[split_idx:].copy()
        
        # Hacer predicciones en el test set
        X_test = test_df[predictor.feature_names]
        preds_prob = predictor.model.predict(X_test)
        test_df["pred_prob"] = preds_prob
        
        # 2. Lógica de Simulación
        capital = tc.initial_capital
        portfolio_history = []
        trades_history = []
        
        wins = 0
        losses = 0
        
        for idx, row in test_df.iterrows():
            # Guardar valor del portafolio en el tiempo
            portfolio_history.append({
                "time": row["datetime"],
                "value": capital
            })
            
            prob = row["pred_prob"]
            target = row["target"]
            
            # Condición de Edge (ventaja)
            # Si predecimos > 55% de subir, y el mercado está al 50% (asumido para simplificar), tenemos un 5% de edge.
            # Aquí usaremos una señal simple: prob > 0.5 + min_edge_threshold
            if prob > (0.5 + tc.min_edge_threshold):
                # Señal de COMPRA
                bet_size = capital * tc.max_position_pct * tc.kelly_fraction
                
                # Si el target fue 1 (subió), ganamos. Si fue 0, perdemos lo apostado.
                # En Polymarket real depende del precio de compra. Asumiremos compra a prob real y venta a 1 o 0.
                if target == 1:
                    ganancia = bet_size * 0.5  # Asumiendo ROI conservador por operación
                    capital += ganancia
                    wins += 1
                    status = "WIN"
                else:
                    capital -= bet_size
                    losses += 1
                    status = "LOSS"
                    ganancia = -bet_size
                    
                trades_history.append({
                    "time": row["datetime"],
                    "market": row.get("question", f"Market {row['market_id']}"),
                    "prob": float(prob),
                    "bet_size": float(bet_size),
                    "pnl": float(ganancia),
                    "status": status
                })

        win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
        
        # Reducir a unos 100 puntos para el gráfico del frontend
        # para no saturar la UI con miles de puntos
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
            },
            "chart_data": portfolio_chart,
            "recent_trades": trades_history[-50:]  # Últimos 50 trades
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


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
