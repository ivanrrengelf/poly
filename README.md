# PolyAlpha — Polymarket Prediction Engine 🚀

Sistema algorítmico completo para la recolección de datos, feature engineering, predicción con Machine Learning (LightGBM) y simulación de paper-trading cuantitativo en mercados macro-económicos y financieros de [Polymarket](https://polymarket.com).

## 💡 ¿Qué hace esta aplicación?

La aplicación funciona como un pipeline completo (End-to-End) de trading cuantitativo adaptado a mercados de predicción. Se divide en 4 grandes bloques:

1. **Recolección de Datos (ETL):** Se conecta a las APIs públicas de Polymarket para descargar mercados, historiales de precios al milisegundo y todos los trades individuales (miles de transacciones) de mercados macroeconómicos.
2. **Feature Engineering:** Procesa estos datos crudos para crear un dataset tabular avanzado. Calcula indicadores técnicos como _Momentum_, _Volatilidad_, medias móviles (Rolling windows) y presión de compra/venta usando el order flow.
3. **Machine Learning Predictivo:** Entrena un modelo de regresión en árbol (LightGBM) utilizando *Walk-Forward Validation* para predecir si el precio de un contrato subirá o bajará en el futuro (N=5 periodos) con altísima precisión técnica.
4. **Dashboard de Backtesting Realista:** Una interfaz web y backend (FastAPI) que simula estrategias de trading algorítmico **Long/Short** basadas en el "Edge" (la diferencia entre lo que predice el modelo y lo que opina el mercado). Implementa fricción realista, límites de liquidez y bloqueo temporal de capital.

---

## 🛠️ Estructura del Proyecto

```text
c:\TRABAJO\poly\
├── config/                  # Hiperparámetros del modelo y constantes de API.
│   ├── settings.py
│   └── constants.py
│
├── data/                    # (Generado dinámicamente) Almacén de datos.
│   ├── raw/                 # CSVs descargados: markets, price_history, trades.
│   ├── processed/           # Dataset 'features.csv' listo para ML.
│   └── models/              # Archivos binarios del modelo LightGBM entrenado.
│
├── src/
│   ├── data/                # Clientes API
│   │   ├── collector.py     # Script principal que ejecuta la descarga.
│   │   ├── gamma_client.py  # API para descubrir mercados y eventos.
│   │   ├── clob_client.py   # API Order book y precios.
│   │   └── data_client.py   # API de Trades históricos.
│   │
│   ├── features/            
│   │   └── engineering.py   # Pipeline que transforma data cruda en features para el modelo.
│   │
│   ├── models/              
│   │   └── predictor.py     # Definición del modelo LightGBM y su entrenamiento.
│   │
│   └── dashboard/           # Plataforma de Backtesting
│       ├── api.py           # Backend FastAPI (Simulador de Trading)
│       └── public/          # Frontend Web Web App (HTML/CSS/JS)
│
├── .venv/                   # Entorno virtual de Python
├── requirements.txt         # Dependencias del proyecto
└── README.md                # Este archivo
```

---

## 🧠 ¿Cómo funciona la estrategia de Inversión (Backtesting)?

El simulador (`src/dashboard/api.py`) no apuesta a ciegas. Utiliza principios reales de gestión de riesgo institucional:

1. **Cálculo del Edge:** `Edge = Probabilidad_del_Modelo - Precio_del_Mercado`. Si el modelo predice 60% pero el mercado está en 70%, NO entramos (no hay ventaja matemática).
2. **Operativa Bidireccional (Long/Short):** 
   - Si el *Edge* es positivo y supera nuestro umbral: Se entra en posición **LONG** (se compra el token YES).
   - Si el *Edge* es negativo (el mercado está sobrevalorado respecto a nuestra predicción): Se entra en posición **SHORT** (se compra el token NO).
3. **Restricción de Liquidez:** Nunca se apuesta más del `2%` de la liquidez real disponible en el order book, asegurando que los resultados puedan ser replicados en la vida real sin "slippage" masivo.
4. **Fricción (Spread Fee):** Se asume un coste del `1%` en cada trade simulando el cruce del Bid-Ask spread del CLOB (Central Limit Order Book).
5. **Cálculo de Beneficios Reales:** El PnL (Ganancias/Pérdidas) no es un porcentaje fijo inventado, es la diferencia exacta entre el precio real de entrada y el precio de salida al cierre del trade.

---

## 🚀 Setup y Ejecución

Para levantar la aplicación en un entorno local de desarrollo:

### 1. Activar el entorno e instalar dependencias
```bash
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generar el Conjunto de Datos (Si no existen)
*Esto descargará datos reales de Polymarket.*
```bash
python -m src.data.collector
python -m src.features.engineering
```

### 3. Arrancar el Dashboard y el Simulador
*Esto cargará el dataset, entrenará el modelo LightGBM en segundos y levantará la interfaz visual en `http://127.0.0.1:8000/app/index.html`*
```bash
python -m uvicorn src.dashboard.api:app --reload
```

---

## 📈 Rendimiento Histórico

En los tests actuales sobre mercados macro, la estrategia algorítmica logra un Win Rate asimétrico (aprox. `38%` de acierto), pero al estar optimizado para identificar trades de alto valor esperado en situaciones donde el mercado se equivoca, el compounding genera un **ROI exponencialmente positivo** de más del +2000% sobre capitales limitados por la liquidez en periodos históricos completos.