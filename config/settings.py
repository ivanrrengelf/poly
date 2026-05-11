from dataclasses import dataclass


@dataclass(frozen=True)
class ModelHyperparams:
    """Hiperparámetros del modelo de predicción (LightGBM)."""

    # Modelo
    n_estimators: int = 500
    learning_rate: float = 0.05
    max_depth: int = 6
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 0.1

    # Entrenamiento
    validation_split: float = 0.2
    early_stopping_rounds: int = 50
    random_seed: int = 42


@dataclass(frozen=True)
class FeatureConfig:
    """Configuración de feature engineering."""

    rolling_windows: tuple = (5, 10, 30, 60)
    lag_periods: tuple = (1, 3, 5, 10, 30)
    price_sampling_interval: str = "1h"
    min_data_points: int = 100


@dataclass(frozen=True)
class TradingConfig:
    """Configuración de paper trading y risk management."""

    min_edge_threshold: float = 0.05       # Edge mínimo para señal (5%)
    kelly_fraction: float = 0.25           # Fracción del Kelly Criterion
    max_position_pct: float = 0.10         # Máx 10% del capital por posición
    max_trade_size_eur: float = 2.0        # Tope absoluto por operación
    execution_fee_pct: float = 0.002       # Coste por ejecución y salida
    slippage_pct: float = 0.001            # Slippage simulado por lado
    holding_period_periods: int = 5        # Horizonte fijo para cerrar posición
    min_entry_price: float = 0.10          # Evita contratos demasiado baratos
    max_entry_price: float = 0.90          # Evita contratos demasiado caros
    max_concurrent_positions: int = 3      # Limita el apilado de posiciones
    min_liquidity_usd: float = 1000.0
    min_volume_24h_usd: float = 500.0
    initial_capital: float = 100.0         # Capital inicial simulado ($100)
