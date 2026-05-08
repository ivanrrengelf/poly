# Poly — Polymarket Prediction Engine

Sistema de análisis y predicción de mercados **macro-económicos y de finanzas** en [Polymarket](https://polymarket.com).

## Qué hace

1. **Recolecta datos** de las APIs públicas de Polymarket (Gamma, CLOB, Data)
2. **Construye features** a partir de series temporales y microestructura del mercado
3. **Entrena un modelo** (LightGBM) para estimar probabilidades de outcomes
4. **Genera señales** de compra/venta comparando la predicción del modelo con el precio de mercado
5. **Simula inversiones** en un dashboard de paper trading

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

## Estructura

```
config/         → Hiperparámetros, constantes, configuración
src/data/       → Clientes API (Gamma, CLOB, Data)
src/features/   → Feature engineering
src/models/     → Modelo de predicción
src/dashboard/  → Dashboard de simulación
tests/          → Tests
```

## APIs de Polymarket

| API | Base URL | Uso |
|-----|----------|-----|
| Gamma | `gamma-api.polymarket.com` | Descubrir mercados y eventos |
| CLOB | `clob.polymarket.com` | Precios, order book, historial |
| Data | `data-api.polymarket.com` | Trades, open interest, holders |

> Todas las lecturas son **sin autenticación**.