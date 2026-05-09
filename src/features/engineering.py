"""Feature engineering para datos de Polymarket.

Transforma datos crudos (precios, trades) en features para el modelo.
Todas las features se calculan con datos pasados (sin data leakage).
"""
import os
import numpy as np
import pandas as pd

from config.settings import FeatureConfig
from src.utils.logger import get_logger

log = get_logger("features")

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

cfg = FeatureConfig()


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carga los CSVs crudos."""
    markets = pd.read_csv(os.path.join(RAW_DIR, "markets.csv"))
    prices = pd.read_csv(os.path.join(RAW_DIR, "price_history.csv"))
    trades_path = os.path.join(RAW_DIR, "trades.csv")
    trades = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()

    log.info(f"Cargados: {len(markets)} mercados, {len(prices)} precios, {len(trades)} trades")
    return markets, prices, trades


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade precios pasados como features (lag_1, lag_3, lag_5, ...)."""
    for lag in cfg.lag_periods:
        df[f"price_lag_{lag}"] = df.groupby("market_id")["price"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade estadísticas de ventana deslizante (media, std, min, max)."""
    for window in cfg.rolling_windows:
        grp = df.groupby("market_id")["price"]

        df[f"rolling_mean_{window}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )
        df[f"rolling_min_{window}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f"rolling_max_{window}"] = grp.transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
    return df


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade momentum y rate of change."""
    grp = df.groupby("market_id")["price"]

    # Diferencia de precio (momentum)
    df["momentum_1"] = grp.diff(1)
    df["momentum_5"] = grp.diff(5)
    df["momentum_10"] = grp.diff(10)

    # Rate of change (% cambio)
    df["roc_1"] = grp.pct_change(1)
    df["roc_5"] = grp.pct_change(5)
    df["roc_10"] = grp.pct_change(10)

    # Aceleración (cambio del momentum)
    df["acceleration"] = df.groupby("market_id")["momentum_1"].diff(1)

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Añade métricas de volatilidad."""
    grp = df.groupby("market_id")["price"]

    # Volatilidad como std de retornos
    returns = grp.pct_change()
    df["volatility_10"] = returns.groupby(df["market_id"]).transform(
        lambda x: x.rolling(10, min_periods=2).std()
    )
    df["volatility_30"] = returns.groupby(df["market_id"]).transform(
        lambda x: x.rolling(30, min_periods=2).std()
    )

    # Rango como % del precio
    for window in [10, 30]:
        high = df.groupby("market_id")["price"].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        low = df.groupby("market_id")["price"].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        df[f"range_pct_{window}"] = (high - low) / (df["price"] + 1e-8)

    return df


def add_position_features(df: pd.DataFrame) -> pd.DataFrame:
    """Features sobre la posición del precio respecto a su rango histórico."""
    for window in [30, 60]:
        high = df.groupby("market_id")["price"].transform(
            lambda x: x.rolling(window, min_periods=1).max()
        )
        low = df.groupby("market_id")["price"].transform(
            lambda x: x.rolling(window, min_periods=1).min()
        )
        # Posición relativa [0, 1] dentro del rango
        df[f"position_{window}"] = (df["price"] - low) / (high - low + 1e-8)

    return df


def add_trade_features(prices_df: pd.DataFrame,
                        trades_df: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivadas de los trades (volumen, presión compra/venta)."""
    if trades_df.empty or "timestamp" not in trades_df.columns:
        log.info("No hay datos de trades, saltando trade features")
        return prices_df

    # Agregar trades por market_id y ventana temporal
    trades_df["timestamp"] = pd.to_numeric(trades_df["timestamp"], errors="coerce")
    trades_df["size"] = pd.to_numeric(trades_df["size"], errors="coerce")

    for market_id, group in prices_df.groupby("market_id"):
        market_trades = trades_df[trades_df["market_id"] == market_id]
        if market_trades.empty:
            continue

        # Para cada punto de precio, contar trades recientes
        for idx, row in group.iterrows():
            ts = row["timestamp"]
            # Trades en la última hora
            recent = market_trades[
                (market_trades["timestamp"] >= ts - 3600) &
                (market_trades["timestamp"] < ts)
            ]
            prices_df.at[idx, "trade_count_1h"] = len(recent)
            prices_df.at[idx, "trade_volume_1h"] = recent["size"].sum()

            if len(recent) > 0:
                buy_vol = recent[recent["side"] == "BUY"]["size"].sum()
                sell_vol = recent[recent["side"] == "SELL"]["size"].sum()
                total = buy_vol + sell_vol
                prices_df.at[idx, "buy_pressure_1h"] = buy_vol / total if total > 0 else 0.5
            else:
                prices_df.at[idx, "buy_pressure_1h"] = 0.5

    return prices_df


def add_target(df: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Añade la variable target: precio continuo futuro en N periodos."""
    df["target"] = df.groupby("market_id")["price"].shift(-horizon)
    return df


def build_features() -> pd.DataFrame:
    """Pipeline completo de feature engineering."""
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    markets, prices, trades = load_raw_data()

    if prices.empty:
        log.warning("No hay datos de precios para construir features")
        return pd.DataFrame()

    # Asegurar orden temporal por mercado
    prices.sort_values(["market_id", "timestamp"], inplace=True)
    prices.reset_index(drop=True, inplace=True)

    log.info("Generando lag features...")
    prices = add_lag_features(prices)

    log.info("Generando rolling features...")
    prices = add_rolling_features(prices)

    log.info("Generando momentum features...")
    prices = add_momentum_features(prices)

    log.info("Generando volatility features...")
    prices = add_volatility_features(prices)

    log.info("Generando position features...")
    prices = add_position_features(prices)

    log.info("Generando trade features...")
    prices = add_trade_features(prices, trades)

    log.info("Generando target...")
    prices = add_target(prices)

    # Merge con metadatos de mercado
    if not markets.empty:
        market_meta = markets[["market_id", "volume", "volume_24h", "liquidity"]].copy()
        market_meta["market_id"] = market_meta["market_id"].astype(str)
        prices["market_id"] = prices["market_id"].astype(str)
        prices = prices.merge(market_meta, on="market_id", how="left",
                              suffixes=("", "_market"))

    # Eliminar filas sin target (últimas N de cada mercado)
    before = len(prices)
    prices.dropna(subset=["target"], inplace=True)
    log.info(f"Eliminadas {before - len(prices)} filas sin target")

    # Guardar
    output_path = os.path.join(PROCESSED_DIR, "features.csv")
    prices.to_csv(output_path, index=False)
    log.info(f"Dataset guardado: {output_path} ({len(prices)} filas, {len(prices.columns)} columnas)")

    # Resumen de features
    feature_cols = [c for c in prices.columns
                    if c not in ["timestamp", "datetime", "token_id", "market_id",
                                 "question", "target"]]
    log.info(f"Features generadas: {len(feature_cols)}")
    log.info(f"Columnas: {feature_cols}")

    return prices


if __name__ == "__main__":
    build_features()
