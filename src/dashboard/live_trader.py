"""Motor de Paper Trading en Tiempo Real para Polymarket.

Este script se ejecuta como un demonio (daemon) en segundo plano.
Descarga datos en vivo, calcula features on-the-fly, realiza predicciones
con el modelo pre-entrenado y gestiona un portfolio virtual en una BD SQLite.
"""
import os
import time
import sqlite3
import pandas as pd
import datetime
from contextlib import closing

from config.settings import TradingConfig
from src.data.gamma_client import GammaClient
from src.data.clob_client import ClobClient
from src.features.engineering import (
    add_lag_features, add_rolling_features,
    add_momentum_features, add_volatility_features,
    add_position_features
)
from src.models.predictor import PolyPredictor
from src.utils.logger import get_logger

log = get_logger("live_trader")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")
DB_PATH = os.path.join(DATA_DIR, "live_portfolio.db")

tc = TradingConfig()


class LiveTrader:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.init_db()
        self.gamma = GammaClient()
        self.clob = ClobClient()
        self.predictor = PolyPredictor()
        
        # Cargar modelo entrenado si existe
        model_path = os.path.join(DATA_DIR, "models", "lightgbm_poly.txt")
        if os.path.exists(model_path):
            self.predictor.load_model()
            self.predictor.feature_names = self.predictor.model.feature_name()
            log.info("LiveTrader: Modelo predictivo cargado correctamente.")
        else:
            log.error(f"LiveTrader: No se encontró el modelo en {model_path}. Ejecuta el backtest o train primero.")

    def init_db(self):
        """Inicializa la base de datos virtual del portfolio."""
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            # Tabla de estado del portfolio (Capital Total y Disponible)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolio (
                    id INTEGER PRIMARY KEY,
                    total_capital REAL,
                    available_capital REAL,
                    updated_at TIMESTAMP
                )
            ''')
            # Insertar capital inicial si está vacía
            cursor.execute('SELECT COUNT(*) FROM portfolio')
            if cursor.fetchone()[0] == 0:
                cursor.execute(
                    'INSERT INTO portfolio (id, total_capital, available_capital, updated_at) VALUES (1, ?, ?, ?)',
                    (tc.initial_capital, tc.initial_capital, datetime.datetime.now().isoformat())
                )
            
            # Tabla de trades activos (esperando resolución de 5 horas)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS active_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    question TEXT,
                    type TEXT,
                    entry_price REAL,
                    predicted_prob REAL,
                    edge REAL,
                    bet_size REAL,
                    liquidity_at_entry REAL,
                    entry_time TIMESTAMP,
                    target_close_time TIMESTAMP
                )
            ''')
            
            # Tabla de historial de trades cerrados
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS historical_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    market_id TEXT,
                    question TEXT,
                    type TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    predicted_prob REAL,
                    edge REAL,
                    bet_size REAL,
                    pnl REAL,
                    roi REAL,
                    entry_time TIMESTAMP,
                    close_time TIMESTAMP
                )
            ''')
            conn.commit()

    def get_portfolio_state(self):
        """Obtiene el capital virtual."""
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT total_capital, available_capital FROM portfolio WHERE id=1')
            return cursor.fetchone()

    def update_portfolio_state(self, total_capital, available_capital):
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'UPDATE portfolio SET total_capital=?, available_capital=?, updated_at=? WHERE id=1',
                (total_capital, available_capital, datetime.datetime.now().isoformat())
            )
            conn.commit()

    async def fetch_live_data(self) -> pd.DataFrame:
        """Descarga historial reciente y precio actual para todos los mercados activos."""
        log.info("Obteniendo mercados activos de Gamma API...")
        # Usar todos los eventos para la prueba en vivo y asegurar que encuentre trades
        events = await self.gamma.get_all_events_paginated(max_pages=2)
        
        # Extraer markets de los eventos
        markets = []
        for e in events:
            markets.extend(e.get("markets", []))
            if len(markets) >= 50:
                break
        markets = markets[:50]
        
        all_prices = []
        for m in markets:
            market_id = m.get("conditionId")
            if not market_id:
                continue
                
            # Extract token IDs properly
            import json
            token_ids_raw = m.get("clobTokenIds", "[]")
            if isinstance(token_ids_raw, str):
                try:
                    token_ids = json.loads(token_ids_raw)
                except Exception:
                    token_ids = []
            else:
                token_ids = token_ids_raw
                
            if not token_ids or len(token_ids) == 0:
                continue
                
            token_id_yes = token_ids[0]
                
            # Obtener historia de Clob
            hist = await self.clob.get_prices_history(token_id_yes, fidelity=500)
            if not hist or "history" not in hist:
                continue
                
            # polymarket prices-history API returns a list in "history"
            data = hist["history"] if isinstance(hist, dict) and "history" in hist else hist
            if not isinstance(data, list) or len(data) == 0:
                continue
                
            df_hist = pd.DataFrame(data)
            if "t" in df_hist.columns and "p" in df_hist.columns:
                df_hist.rename(columns={"t": "timestamp", "p": "price"}, inplace=True)
                
            df_hist["market_id"] = token_id_yes
            df_hist["question"] = m.get("question", f"Market {market_id}")
            df_hist["liquidity"] = float(m.get("liquidity", 1000) or 1000)
            
            # Asegurar orden temporal
            df_hist["timestamp"] = pd.to_numeric(df_hist["timestamp"])
            df_hist.sort_values("timestamp", inplace=True)
            all_prices.append(df_hist)
            
        if not all_prices:
            return pd.DataFrame()
            
        return pd.concat(all_prices, ignore_index=True)

    def calculate_live_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica engineering a la ventana de datos para sacar la fila actual."""
        log.info(f"Calculando features en vivo para {len(df['market_id'].unique())} mercados...")
        
        df = add_lag_features(df)
        df = add_rolling_features(df)
        df = add_momentum_features(df)
        df = add_volatility_features(df)
        df = add_position_features(df)
        
        # Llenar NAs hacia atrás (backfill) para mercados nuevos que no tienen 60 periodos de historia
        # para que no se borren por culpa de las medias móviles largas.
        df.bfill(inplace=True)
        df.dropna(inplace=True)
        
        # Quedarnos solo con la ÚLTIMA fila de cada mercado (la más reciente)
        latest_rows = df.groupby("market_id").tail(1).copy()
        return latest_rows

    async def execute_live_predictions(self):
        """Pipeline principal: Predice en vivo y abre posiciones."""
        df = await self.fetch_live_data()
        if df.empty:
            log.warning("LiveTrader: No hay suficientes datos para operar.")
            return

        latest_features = self.calculate_live_features(df)
        
        if latest_features.empty:
            log.warning("LiveTrader: Tras calcular features, el dataframe quedó vacío (faltan datos históricos suficientes).")
            return
            
        # Preparar X para predecir
        missing_cols = set(self.predictor.feature_names) - set(latest_features.columns)
        for c in missing_cols:
            latest_features[c] = 0.0 # Llenar columnas faltantes si las hay (ej. trades pressure)
            
        X = latest_features[self.predictor.feature_names]
        latest_features["pred_diff"] = self.predictor.model.predict(X)

        total_cap, avail_cap = self.get_portfolio_state()
        new_avail_cap = avail_cap
        
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            
            for idx, row in latest_features.iterrows():
                if new_avail_cap < 1.0:
                    break # Sin capital
                    
                market_id = row["market_id"]
                pred_diff = row["pred_diff"]
                precio_actual = row["price"]
                precio_esperado_absoluto = precio_actual + pred_diff
                
                try:
                    book = await self.clob.get_book(market_id)
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if not bids or not asks:
                        continue
                    
                    best_bid = float(bids[0]["price"])
                    best_ask = float(asks[0]["price"])
                except Exception as e:
                    log.error(f"Error fetching book for {market_id}: {e}")
                    continue
                
                # Cálculo de Edge real vs el libro de órdenes
                edge_long = precio_esperado_absoluto - best_ask
                edge_short = best_bid - precio_esperado_absoluto
                
                edge = 0
                is_long = False
                order_price = 0
                
                # Evaluar qué lado del mercado tiene mayor ventaja y configurar la orden límite
                if edge_long > edge_short and edge_long > tc.min_edge_threshold:
                    edge = edge_long
                    is_long = True
                    order_price = best_bid + 0.01 if best_bid + 0.01 < best_ask else best_bid
                elif edge_short > edge_long and edge_short > tc.min_edge_threshold:
                    edge = edge_short
                    is_long = False
                    order_price = best_ask - 0.01 if best_ask - 0.01 > best_bid else best_ask
                
                # Condición de entrada
                if edge > tc.min_edge_threshold:
                    max_bet = total_cap * tc.max_position_pct * tc.kelly_fraction
                    liquidity = float(row.get("liquidity", 1000))
                    liquidity_limit = liquidity * 0.02
                    
                    bet_size = min(max_bet, liquidity_limit, new_avail_cap)
                    
                    if bet_size >= 1.0:
                        # Ejecutar orden virtual
                        new_avail_cap -= bet_size
                        now = datetime.datetime.now()
                        target_close = now + datetime.timedelta(hours=5)
                        
                        log.info(f"🚀 OPEN VIRTUAL LIMIT ORDER: {'LONG' if is_long else 'SHORT'} on {row['question'][:30]}... Expected Price: {precio_esperado_absoluto:.3f} Edge: {edge*100:.1f}% Bet: ${bet_size:.2f} @ {order_price:.3f}")
                        
                        cursor.execute('''
                            INSERT INTO active_trades 
                            (market_id, question, type, entry_price, predicted_prob, edge, bet_size, liquidity_at_entry, entry_time, target_close_time)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            market_id, row["question"], "LONG" if is_long else "SHORT",
                            float(order_price), float(precio_esperado_absoluto), float(edge), float(bet_size), float(liquidity),
                            now.isoformat(), target_close.isoformat()
                        ))
            
            conn.commit()
            
        if new_avail_cap != avail_cap:
            self.update_portfolio_state(total_cap, new_avail_cap)

    async def manage_open_trades(self):
        """Revisa todos los trades activos. Cierra si tocan TP, SL o expiran."""
        now = datetime.datetime.now()
        total_cap, avail_cap = self.get_portfolio_state()
        
        with closing(sqlite3.connect(DB_PATH)) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM active_trades')
            active_trades = cursor.fetchall()
            
            if not active_trades:
                return
                
            columns = [desc[0] for desc in cursor.description]
            
            for trade in active_trades:
                t = dict(zip(columns, trade))
                target_time = datetime.datetime.fromisoformat(t["target_close_time"])
                is_expired = now >= target_time
                
                # Para cerrar, obtenemos el precio *actual en vivo* del CLOB
                try:
                    book = await self.clob.get_book(t["market_id"])
                    bids = book.get("bids", [])
                    asks = book.get("asks", [])
                    if bids and asks:
                        current_price = (float(bids[0]["price"]) + float(asks[0]["price"])) / 2
                    elif bids: current_price = float(bids[0]["price"])
                    elif asks: current_price = float(asks[0]["price"])
                    else: current_price = 0.5 # fallback
                except Exception as e:
                    log.error(f"Error fetching current price for {t['market_id']}: {e}")
                    current_price = t["entry_price"] # Cierre neutro si falla
                
                # Calcular PnL Real
                price_in = t["entry_price"]
                price_out = current_price
                
                if t["type"] == "LONG":
                    roi = (price_out - price_in) / price_in if price_in > 0 else 0
                else: # SHORT (comprar NO)
                    roi = (price_in - price_out) / (1 - price_in) if (1 - price_in) > 0 else 0
                    
                # Aplicar spread fee (fricción real al salir/entrar)
                roi -= 0.01 
                
                # Check dynamic risk management
                close_reason = None
                if roi >= tc.take_profit_pct:
                    close_reason = f"TAKE-PROFIT (+{roi*100:.1f}%)"
                elif roi <= tc.stop_loss_pct:
                    close_reason = f"STOP-LOSS ({roi*100:.1f}%)"
                elif is_expired:
                    close_reason = "EXPIRED (5h)"
                    
                if close_reason:
                    log.info(f"Cerrando trade en {t['question'][:30]}... Motivo: {close_reason}")
                    ganancia = t["bet_size"] * roi
                    
                    # Devolver capital bloqueado + ganancias al portfolio
                    avail_cap += (t["bet_size"] + ganancia)
                    total_cap += ganancia
                    
                    # Guardar en histórico
                    cursor.execute('''
                        INSERT INTO historical_trades 
                        (market_id, question, type, entry_price, exit_price, predicted_prob, edge, bet_size, pnl, roi, entry_time, close_time)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        t["market_id"], t["question"], t["type"], float(price_in), float(price_out),
                        t["predicted_prob"], t["edge"], t["bet_size"], float(ganancia), float(roi),
                        t["entry_time"], now.isoformat()
                    ))
                    
                    # Eliminar de activos
                    cursor.execute('DELETE FROM active_trades WHERE id=?', (t["id"],))
                    
                    log.info(f"✅ Trade Cerrado. PnL: ${ganancia:.2f}")
                
            conn.commit()
            
        self.update_portfolio_state(total_cap, avail_cap)

    async def run_cycle(self):
        """Ejecuta un ciclo completo del bot."""
        log.info("=== INICIANDO CICLO DE LIVE TRADING ===")
        await self.manage_open_trades()
        await self.execute_live_predictions()
        log.info("=== CICLO COMPLETADO ===")


async def main():
    trader = LiveTrader()
    CYCLE_MINUTES = 5
    
    log.info(f"INICIANDO BOT EN PRODUCCION (Ciclos cada {CYCLE_MINUTES} minutos)")
    while True:
        try:
            await trader.run_cycle()
        except Exception as e:
            log.error(f"Error crítico en el ciclo: {e}")
            import traceback
            log.error(traceback.format_exc())
            
        log.info(f"Durmiendo {CYCLE_MINUTES} minutos hasta el siguiente escaneo...")
        await asyncio.sleep(CYCLE_MINUTES * 60)

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
