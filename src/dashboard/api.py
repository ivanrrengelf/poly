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
        
        # Re-crear future_price para el cálculo realista del PnL en simulación
        # Hacerlo sólo en el set de prueba para evitar data leakage
        test_df = test_df.sort_values(["market_id", "timestamp"])
        test_df["future_price"] = test_df.groupby("market_id")["price"].shift(-5)
        
        # Hacer predicciones en el test set
        X_test = test_df[predictor.feature_names]
        preds_prob = predictor.model.predict(X_test)
        test_df["pred_prob"] = preds_prob
        
        # 2. Lógica de Simulación Restringida (Realista)
        capital_total = tc.initial_capital
        capital_disponible = tc.initial_capital
        
        portfolio_history = []
        trades_history = []
        active_trades = []
        
        wins = 0
        losses = 0
        
        for idx, row in test_df.iterrows():
            current_time = pd.to_numeric(row["timestamp"]) if "timestamp" in row else idx
            
            # 2.1 Liberar trades expirados (5 periodos ~ 5 horas * 3600)
            remaining_trades = []
            for t in active_trades:
                if current_time >= t["exit_time"]:
                    # Cerrar trade
                    capital_disponible += (t["bet_size"] + t["pnl"])
                    capital_total += t["pnl"]
                    
                    if t["pnl"] > 0:
                        wins += 1
                    elif t["pnl"] < 0:
                        losses += 1
                        
                    trades_history.append(t["trade_log"])
                else:
                    remaining_trades.append(t)
            active_trades = remaining_trades
            
            # Guardar valor del portafolio en el tiempo
            portfolio_history.append({
                "time": row["datetime"],
                "value": capital_total
            })
            
            price = row["price"]
            future_price = row.get("future_price")
            if pd.isna(future_price):
                continue
                
            prob = row["pred_prob"]
            edge = prob - price
            
            # Condición de Edge (ventaja) y capital suficiente
            if abs(edge) > tc.min_edge_threshold and capital_disponible > 1.0:
                is_long = edge > 0 
                
                # Tamaño base
                max_bet = capital_total * tc.max_position_pct * tc.kelly_fraction
                
                # Restricción de Liquidez (Max 2% de liquidez actual)
                # OJO: en features.csv se puede llamar liquidity_market
                liquidity = row.get("liquidity", row.get("liquidity_market", 1000))
                if pd.isna(liquidity) or liquidity <= 0:
                    liquidity = 1000
                liquidity_limit = liquidity * 0.02
                
                # Tamaño real (el mínimo de las 3 restricciones)
                bet_size = min(max_bet, liquidity_limit, capital_disponible)
                
                if bet_size < 1.0:
                    continue # Muy poco capital o liquidez
                
                # Retorno crudo
                if is_long:
                    roi = (future_price - price) / price if price > 0 else 0
                else:
                    roi = (price - future_price) / (1 - price) if (1 - price) > 0 else 0
                
                # Restar fricción (1% por spread)
                spread_fee = 0.01 
                roi -= spread_fee
                
                ganancia = bet_size * roi
                status = "WIN" if ganancia > 0 else "LOSS"
                
                # Bloquear capital
                capital_disponible -= bet_size
                
                trade_log = {
                    "time": row["datetime"],
                    "market": row.get("question", f"Market {row['market_id']}"),
                    "prob": float(prob),
                    "edge": float(edge * 100),
                    "type": "LONG" if is_long else "SHORT",
                    "bet_size": float(bet_size),
                    "pnl": float(ganancia),
                    "status": status
                }
                
                active_trades.append({
                    "exit_time": current_time + (5 * 3600),
                    "bet_size": bet_size,
                    "pnl": ganancia,
                    "trade_log": trade_log
                })

        # Al final de la serie, forzar el cierre de todos los trades activos
        for t in active_trades:
            capital_total += t["pnl"]
            if t["pnl"] > 0:
                wins += 1
            elif t["pnl"] < 0:
                losses += 1
            trades_history.append(t["trade_log"])

        capital = capital_total

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

    except Exception as e:
        log.error(f"Error en simulación: {e}")
        return {"error": str(e)}


@app.get("/api/simulation")
async def get_simulation():
    """Endpoint que devuelve los resultados del backtest."""
    return run_simulation()


@app.get("/api/live")
async def get_live_portfolio():
    """Endpoint que devuelve el estado actual del Paper Trader en Producción."""
    import sqlite3
    from contextlib import closing
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "live_portfolio.db")
    
    if not os.path.exists(db_path):
        return {"error": "El Live Trader no ha sido inicializado. Ejecuta src/dashboard/live_trader.py primero."}
        
    try:
        with closing(sqlite3.connect(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Obtener portfolio
            cursor.execute('SELECT * FROM portfolio WHERE id=1')
            portfolio = dict(cursor.fetchone())
            
            # Obtener trades activos
            cursor.execute('SELECT * FROM active_trades ORDER BY entry_time DESC')
            active = [dict(r) for r in cursor.fetchall()]
            
            # Obtener histórico
            cursor.execute('SELECT * FROM historical_trades ORDER BY close_time DESC LIMIT 50')
            history = [dict(r) for r in cursor.fetchall()]
            
            # Calcular Win Rate histórico
            wins = sum(1 for t in history if t['pnl'] > 0)
            win_rate = (wins / len(history) * 100) if history else 0
            
            return {
                "metrics": {
                    "initial_capital": 100.0, # Asumido por TradingConfig default
                    "final_capital": portfolio["total_capital"],
                    "available_capital": portfolio["available_capital"],
                    "roi_pct": ((portfolio["total_capital"] - 100.0) / 100.0) * 100,
                    "total_trades": len(history),
                    "active_trades_count": len(active),
                    "win_rate": win_rate,
                    "updated_at": portfolio["updated_at"]
                },
                "active_trades": active,
                "recent_trades": history
            }
    except Exception as e:
        log.error(f"Error reading live DB: {e}")
        return {"error": str(e)}


@app.get("/")
async def root():
    """Redirige al dashboard."""
    return {"message": "API Running. Go to /app/index.html to view the dashboard."}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

