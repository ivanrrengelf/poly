"""Fixtures centralizadas para todos los tests.

Proporciona datos mockeados para:
- Mercados Gamma
- Historial de precios CLOB
- Trades históricos
- Respuestas HTTP mockeadas
"""
import json
from datetime import datetime

import pytest
import responses
import pandas as pd

from src.data.gamma_client import GammaClient
from src.data.clob_client import ClobClient
from src.data.data_client import DataClient


# ============================================================================
# Fixtures de Datos
# ============================================================================


@pytest.fixture
def sample_gamma_event() -> dict:
    """Evento de Gamma API con mercados embebidos."""
    return {
        "id": "0x123",
        "title": "Will Bitcoin reach $100k by Dec 2024?",
        "slug": "bitcoin-100k-dec-2024",
        "tags": [
            {"id": "120", "label": "Finance", "slug": "finance"},
            {"id": "21", "label": "Crypto", "slug": "crypto"},
        ],
        "active": True,
        "closed": False,
        "markets": [
            {
                "id": "0xMARKET1",
                "conditionId": "0xCOND1",
                "question": "Will Bitcoin reach $100k by Dec 2024?",
                "clobTokenIds": [
                    "123456789012345678901234567890123456789012345678901234567890123456",
                    "234567890123456789012345678901234567890123456789012345678901234567",
                ],
                "outcomes": ["Yes", "No"],
                "outcomePrices": [0.65, 0.35],
                "volume": 1000000,
                "volume24hr": 50000,
                "liquidity": 100000,
                "endDate": "2024-12-31",
                "createdAt": "2024-01-01T00:00:00Z",
            }
        ],
    }


@pytest.fixture
def sample_clob_prices() -> dict:
    """Historial de precios desde CLOB API."""
    return {
        "history": [
            {"t": 1700000000, "p": 0.60},
            {"t": 1700001000, "p": 0.62},
            {"t": 1700002000, "p": 0.61},
            {"t": 1700003000, "p": 0.63},
            {"t": 1700004000, "p": 0.65},
        ]
    }


@pytest.fixture
def sample_trades() -> dict:
    """Trades históricos desde Data API."""
    return {
        "data": [
            {
                "timestamp": 1700000100,
                "market": "0xMARKET1",
                "side": "BUY",
                "size": 100,
                "price": 0.60,
            },
            {
                "timestamp": 1700000500,
                "market": "0xMARKET1",
                "side": "SELL",
                "size": 50,
                "price": 0.62,
            },
            {
                "timestamp": 1700001500,
                "market": "0xMARKET1",
                "side": "BUY",
                "size": 75,
                "price": 0.61,
            },
        ]
    }


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Minimal features DataFrame para testing."""
    return pd.DataFrame(
        {
            "timestamp": [1700000000 + i * 1000 for i in range(50)],
            "datetime": [
                datetime.fromtimestamp(1700000000 + i * 1000) for i in range(50)
            ],
            "market_id": ["0xMARKET1"] * 50,
            "price": [0.50 + i * 0.001 for i in range(50)],
            "token_id": ["token123"] * 50,
            "price_lag_1": [0.50] + [0.50 + i * 0.001 for i in range(49)],
            "rolling_mean_5": [0.50 + i * 0.001 for i in range(50)],
            "momentum_1": [0.001] * 50,
            "volatility_10": [0.02] * 50,
            "trade_count_1h": [5] * 50,
            "volume": [100000] * 50,
            "target": [1 if i % 2 == 0 else 0 for i in range(50)],
        }
    )


# ============================================================================
# Fixtures para HTTP Mocking
# ============================================================================


@pytest.fixture
def mock_gamma_api(sample_gamma_event) -> responses.RequestsMock:
    """Mockea la Gamma API."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "https://gamma-api.polymarket.com/events",
            json=[sample_gamma_event],
            status=200,
        )
        rsps.add(
            responses.GET,
            "https://gamma-api.polymarket.com/tags",
            json=[
                {"id": "120", "label": "Finance", "slug": "finance"},
                {"id": "21", "label": "Crypto", "slug": "crypto"},
            ],
            status=200,
        )
        yield rsps


@pytest.fixture
def mock_clob_api(sample_clob_prices) -> responses.RequestsMock:
    """Mockea la CLOB API."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "https://clob.polymarket.com/prices-history",
            json=sample_clob_prices,
            status=200,
        )
        rsps.add(
            responses.GET,
            "https://clob.polymarket.com/price",
            json={"price": 0.65},
            status=200,
        )
        rsps.add(
            responses.GET,
            "https://clob.polymarket.com/midpoint",
            json={"midpoint": 0.65},
            status=200,
        )
        rsps.add(
            responses.GET,
            "https://clob.polymarket.com/spread",
            json={"bid": 0.64, "ask": 0.66},
            status=200,
        )
        yield rsps


@pytest.fixture
def mock_data_api(sample_trades) -> responses.RequestsMock:
    """Mockea la Data API."""
    with responses.RequestsMock() as rsps:
        rsps.add(
            responses.GET,
            "https://data-api.polymarket.com/trades",
            json=sample_trades,
            status=200,
        )
        rsps.add(
            responses.GET,
            "https://data-api.polymarket.com/oi",
            json={"oi": 500000},
            status=200,
        )
        yield rsps


# ============================================================================
# Fixtures para Clientes
# ============================================================================


@pytest.fixture
async def gamma_client():
    """Crea un GammaClient para testing."""
    client = GammaClient()
    yield client
    await client.close()


@pytest.fixture
async def clob_client():
    """Crea un ClobClient para testing."""
    client = ClobClient()
    yield client
    await client.close()


@pytest.fixture
async def data_client():
    """Crea un DataClient para testing."""
    client = DataClient()
    yield client
    await client.close()


# ============================================================================
# Configuración de Pytest
# ============================================================================


def pytest_configure(config):
    """Hook de pytest para configuración global."""
    config.addinivalue_line(
        "markers", "integration: tests que requieren acceso a APIs reales"
    )
    config.addinivalue_line(
        "markers", "slow: tests que tardan más de 5 segundos"
    )
