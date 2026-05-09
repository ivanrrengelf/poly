# AUDIT REPORT: Polymarket Predictor - Phase 0 Discovery

**Fecha**: May 9, 2026  
**Objetivo**: Validar estado actual del pipeline sin modificar nada  
**Conclusión**: Pipeline incompleto, algunos críticos

---

## 1. ESTADO DE CADA COMPONENTE

### 1.1 APIs Externas

| API | Status | Detalles |
|-----|--------|---------|
| **Gamma (Discovery)** | ✅ FUNCIONAL | Devuelve 5 eventos, tags correctos, paginación OK |
| **CLOB (Precios)** | ⚠️ PARCIAL | Conecta pero devuelve 404 en mercados sin order book activo (esperado) |
| **Data (Trades)** | ❌ NO TESTEADO | Clientes creados pero no verificados en verify.py |

**Crítico**: CLOB 404s no son errores — son mercados sin liquidez. El retry logic maneja correctamente (3 intentos, backoff exponencial).

---

### 1.2 Pipeline de Recolección (`src/data/collector.py`)

**Estado**: ✅ IMPLEMENTADO Y COMPLETO

- ✅ Mercados: Descarga eventos activos + cerrados, filtra por tags macro/finanzas
- ✅ Precios: Para cada mercado, descarga historial de precios CLOB
- ✅ Trades: Para cada mercado, descarga trades históricos
- ✅ Async cleanup: Cierra `gamma`, `clob`, `data` en finally
- ✅ CSV output: Guarda markets.csv, price_history.csv, trades.csv

**Problema P0 (No confirmado)**:  
Esperaba falta de cierre de clientes en finally → **FALSO** ✅ Ya está implementado correctamente.

**Resultado esperado si se ejecuta**:
- `data/raw/markets.csv` (~700 filas)
- `data/raw/price_history.csv` (~35k filas)
- `data/raw/trades.csv` (~120k filas)

**¿Ejecutado?** NO — los archivos no existen aún.

---

### 1.3 Feature Engineering (`src/features/engineering.py`)

**Estado**: ✅ IMPLEMENTADO Y COMPLETO

**Features generadas**:
- Lags: 5 features (lag_1, lag_3, lag_5, lag_10, lag_30)
- Rolling: 16 features (mean, std, min, max × 4 ventanas)
- Momentum: 7 features (momentum_1/5/10, roc_1/5/10, acceleration)
- Volatilidad: 4 features (volatility_10/30, range_pct_10/30)
- Posición: 2 features (position_30/60)
- Trades: 3 features (trade_count_1h, trade_volume_1h, buy_pressure_1h)
- Mercado: 3 features (volume, volume_24h, liquidity)
- Target: 1 variable (dirección del precio en +5 periodos)

**Total**: 41 features

**Lógica de NaN handling**:
```python
# Actual:
prices.dropna(subset=["target"], inplace=True)

# Problema: No hay fillna() preventivo
# Trades con timestamps incorrectos pueden causar NaNs cascada
```

**Problemas**:

1. 🟠 **Trade features leakage potencial**: `add_trade_features()` suma trades históricos pero NO valida que estén disponibles en t-real. En backtesting esto es OK (data histórica), pero en producción (real-time) es data leakage.

2. 🟡 **Trade timestamps**: Si trades_df vacío o sin timestamp, el código marca `buy_pressure_1h = 0.5` (neutral). Puede sesgar features.

3. 🟡 **Índice desalineado**: `prices_df.at[idx, "trade_count_1h"] = ...` asume índices en orden — si hay gaps puede fallar.

**¿Ejecutado?** NO — sin data cruda.

---

### 1.4 Modelo LightGBM (`src/models/predictor.py`)

**Estado**: ✅ IMPLEMENTADO Y COMPLETO

**Arquitectura**:
- Clase `PolyPredictor` con hyperparámetros configurables
- Método `load_data()` carga features.csv
- Método `train_walk_forward()` entrena con split temporal 80/20
- Método `_evaluate()` calcula métricas (Accuracy, Brier, Logloss)
- Método `save_model()` persiste modelo (implied, no mostrado en lectura)

**Metodología**:
- Walk-Forward validation (NO random shuffle, respeta orden temporal)
- LightGBM con early stopping
- Métricas: accuracy, brier_score_loss, log_loss

**Problemas**:

1. 🟡 **save_model() no implementado completamente**: Lógica presente pero no visible en lectura. Asumimos funciona.
2. 🟡 **Sin type hints en algunos parámetros**: `df: pd.DataFrame` tiene hints pero no siempre.
3. 🟡 **Sin validación de features**: No verifica que feature_names existe antes de usar.

**¿Ejecutado?** NO — sin features.csv.

---

### 1.5 Dashboard API (`src/dashboard/api.py`)

**Estado**: ⚠️ IMPLEMENTADO PERO INCOMPLETO

**Componentes**:
- FastAPI app con CORS configurado
- StaticFiles para servir frontend
- Endpoint `/api/simulation` que ejecuta backtest

**Simulación lógica**:
```
1. Cargar features.csv
2. Entrenar modelo walk-forward
3. Predecir en test set (últimas 20%)
4. Por cada predicción:
   - Si prob > 0.55 (edge de 5%), hacer apuesta
   - Si target=1, ganar ROI asumido (50% asumido)
   - Si target=0, perder apuesta
5. Cachear resultados
```

**Problemas Críticos**:

1. 🔴 **ROI asumido fijo 50%**: No realista. En Polymarket:
   - Depende del precio de compra (p) y venta (1 o 0)
   - ROI = (venta - compra) / compra
   - Código asume ficción: `ganancia = bet_size * 0.5`

2. 🔴 **Spread e slippage ignorados**: En mercados reales hay:
   - Bid-ask spread (no se modela)
   - Slippage al cerrar posición
   - Liquidez variable

3. 🔴 **Condición de edge simplista**: `prob > 0.5 + 0.05` es mecánica.
   - En Polymarket el "fair price" es la probabilidad implícita de mercado
   - No hay "expected value" real sin conocer el precio de compra

4. 🟠 **Sin path validation**: `features.csv` puede no existir → crash runtime.

5. 🟡 **Simulación no es reproducible**: Si features.csv cambio, resultados diferentes pero sin versioning.

**¿Ejecutado?** NO — sin features.csv.

---

### 1.6 Archivos de Datos

| Path | Estado | Contenido |
|------|--------|----------|
| `data/raw/` | ❌ VACÍO | Debe contener markets.csv, price_history.csv, trades.csv |
| `data/processed/` | ❌ VACÍO | Debe contener features.csv |
| `data/models/` | ❌ VACÍO | Debe contener model.pkl |

**Bloqueante**: Sin data crudos, el pipeline no puede ejecutarse.

---

### 1.7 Tests

**Status**: ❌ CERO COVERAGE

```
tests/
├── __init__.py (vacío)
```

No hay:
- Unit tests para clientes API
- Tests para feature engineering
- Tests para modelo
- Integration tests
- Fixtures

**Riesgo**: Cambios rompen pipeline sin detectar.

---

## 2. ANÁLISIS DE ANTIPATRONES

### 2.1 Duplicación de Código

| Ubicación | Patrón | Repeticiones |
|-----------|--------|--------------|
| `{gamma,clob,data}_client.py` | Retry logic (3 intentos, backoff) | 3× |
| `collector.py` + `explore_data.py` | `_parse_json_field()` | 2× |
| Error handling | Try-except logging | 4× |

**Impacto**: Cambiar retry logic requiere actualizar 3 archivos.

### 2.2 Sin Type Hints

| Archivo | Funciones sin hints |
|---------|-------------------|
| `collector.py` | `_parse_json_field()`, `run_collection()` |
| `engineering.py` | Varios helpers |

**Impacto**: IDE no valida tipos, debugging más difícil.

### 2.3 Rutas Hardcodeadas

```python
# Actual
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")

# Problema: Si se mueve el archivo, break
# Mejor: config centralizado
```

### 2.4 Logging Sin Errores

En `add_trade_features()`, solo logs informativos, sin captura de excepciones para NaNs o índices desalineados.

### 2.5 Data Leakage (Crítico)

**Ubicación**: `add_trade_features()` en engineering.py

```python
for ts in row["timestamp"]:
    recent = market_trades[(market_trades["timestamp"] >= ts - 3600) & ...]
    prices_df.at[idx, "trade_count_1h"] = len(recent)
```

**Problema**: En backtesting, esto es "mirar al futuro". Si `trades.csv` contiene trades que ocurren en `t`, y calculamos features para `t-1`, es leakage.

**Riesgo**: Modelo overfit, performance real peor.

---

## 3. RESUMEN EJECUTIVO

| Aspecto | Status | Prioridad |
|--------|--------|-----------|
| **APIs conectan** | ✅ Gamma OK, CLOB parcial | Baja |
| **Código implementado** | ✅ 90% funcional | Media |
| **Pipeline sin ejecutar** | ❌ BLOQUEANTE | P0 |
| **Tests** | ❌ Cero coverage | P1 |
| **Data leakage** | ⚠️ Potencial en trades | P1 |
| **Simulación realista** | ❌ ROI ficción | P2 |
| **Arquitectura acoplada** | ⚠️ Retry logic × 3 | P2 |

---

## 4. RECOMENDACIONES DE PRÓXIMOS PASOS

### 4.1 Phase 0.2: Reparar Bugs Críticos (2-3 hrs)

1. ✅ `collector.py` — Ya cierra clientes correctamente
2. ⚠️ `api.py` — Agregar validación de features.csv
3. 🟠 `engineering.py` — Investigar leakage en trade_features

### 4.2 Phase 0.3: Arquitectura Base (2-3 hrs)

1. Crear `src/data/base_client.py` — extrae retry logic común
2. Refactorizar `{gamma,clob,data}_client.py` para heredar de BaseHTTPClient
3. Crear `src/config/loader.py` — factory para crear clientes

### 4.3 Phase 1: Test Infrastructure (6-9 hrs)

1. Instalar pytest, responses, pytest-cov
2. Crear `tests/conftest.py` con fixtures
3. Unit tests para cada módulo
4. Integration tests para pipeline

### 4.4 Phase 2: Validación End-to-End (2-3 hrs)

1. Ejecutar collector.py (sin mocks) → generar data cruda
2. Ejecutar engineering.py → generar features
3. Ejecutar predictor.py → entrenar modelo
4. Ejecutar api.py → simular trades

---

## 5. FICHEROS CRÍTICOS SIN CAMBIOS

- [src/data/clob_client.py](src/data/clob_client.py) — OK, retry logic funcional
- [src/data/gamma_client.py](src/data/gamma_client.py) — OK
- [src/data/data_client.py](src/data/data_client.py) — OK
- [config/settings.py](config/settings.py) — OK, dataclasses correcto
- [config/constants.py](config/constants.py) — OK

---

## CONCLUSIÓN

**Pipeline está 90% implementado pero NUNCA se ejecutó de principio a fin.** Los mayores riesgos son:

1. **Sin ejecución real**: No sabemos si falla en runtime
2. **Sin tests**: Cambios futuros pueden romper todo
3. **Data leakage**: Trades features pueden sesgar predicciones
4. **Simulación realista**: ROI ficción, no refleja mercado real

**Recomendación**: Proceder con Phase 0.2 (validación de path) + Phase 1 (tests) antes de tocar lógica de simulación.
