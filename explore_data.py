"""Explora la estructura de datos reales de las APIs para diseñar features."""
import asyncio
import json
from src.data.gamma_client import GammaClient
from src.data.clob_client import ClobClient
from src.data.data_client import DataClient


async def main():
    gamma = GammaClient()
    clob = ClobClient()
    data = DataClient()

    # 1. Buscar mercados macro/finanzas activos
    print("=" * 60)
    print("Buscando mercados macro/finanzas...")
    print("=" * 60)
    events = await gamma.get_events(limit=50, active=True, closed=False)

    # Filtrar manualmente por tags relevantes
    macro_keywords = {"economics", "federal reserve", "inflation", "interest rates",
                      "gdp", "earnings", "stocks", "financial markets", "business",
                      "economy", "fed", "stock market", "finance"}

    macro_events = []
    for e in events:
        tags = e.get("tags", [])
        tag_labels = []
        for t in tags:
            if isinstance(t, dict):
                tag_labels.append(t.get("label", "").lower())
            elif isinstance(t, str):
                tag_labels.append(t.lower())

        if any(kw in label for label in tag_labels for kw in macro_keywords):
            macro_events.append(e)

    print(f"Total eventos: {len(events)}, Macro/finanzas: {len(macro_events)}")
    print()

    # Si no hay macro, usar los primeros que tengan mercados con order book
    target_events = macro_events if macro_events else events[:10]

    # 2. Para cada evento, explorar estructura de mercado
    for e in target_events[:3]:
        print(f"EVENTO: {e.get('title', '?')}")
        markets = e.get("markets", [])
        print(f"  Mercados: {len(markets)}")

        for m in markets[:2]:
            print(f"\n  MERCADO: {m.get('question', '?')}")
            print(f"    ID: {m.get('id', '?')}")
            print(f"    conditionId: {m.get('conditionId', '?')}")

            # Outcomes y precios
            outcomes = m.get("outcomes", "[]")
            if isinstance(outcomes, str):
                outcomes = json.loads(outcomes)
            prices = m.get("outcomePrices", "[]")
            if isinstance(prices, str):
                prices = json.loads(prices)
            print(f"    Outcomes: {outcomes}")
            print(f"    Prices: {prices}")

            # Token IDs
            token_ids = m.get("clobTokenIds", "[]")
            if isinstance(token_ids, str):
                token_ids = json.loads(token_ids)
            print(f"    Token IDs: {token_ids}")

            # Volumen y liquidez
            print(f"    Volume: {m.get('volume', '?')}")
            print(f"    Volume24hr: {m.get('volume24hr', '?')}")
            print(f"    Liquidity: {m.get('liquidity', '?')}")
            print(f"    enableOrderBook: {m.get('enableOrderBook', '?')}")

            # 3. Intentar obtener historial de precios del CLOB
            if token_ids and m.get("enableOrderBook"):
                token_id = token_ids[0]
                try:
                    history = await clob.get_prices_history(token_id, fidelity=10)
                    print(f"\n    HISTORIAL DE PRECIOS (primeros 3 puntos):")
                    if isinstance(history, dict):
                        print(f"    Keys: {list(history.keys())}")
                        # Mostrar la estructura completa del primer item
                        hist_data = history.get("history", history)
                        if isinstance(hist_data, list) and hist_data:
                            for h in hist_data[:3]:
                                print(f"      {h}")
                    elif isinstance(history, list) and history:
                        for h in history[:3]:
                            print(f"      {h}")
                    else:
                        print(f"    Raw: {history}")
                except Exception as ex:
                    print(f"    Error historial: {ex}")

                # 4. Intentar obtener trades
                try:
                    trades = await data.get_trades(market=m.get("conditionId"), limit=5)
                    print(f"\n    TRADES (primeros 3):")
                    if isinstance(trades, list):
                        for tr in trades[:3]:
                            print(f"      {tr}")
                    elif isinstance(trades, dict):
                        print(f"    Keys: {list(trades.keys())}")
                        trade_list = trades.get("data", trades.get("trades", []))
                        for tr in trade_list[:3]:
                            print(f"      {tr}")
                except Exception as ex:
                    print(f"    Error trades: {ex}")

                # 5. Spread y midpoint
                try:
                    spread = await clob.get_spread(token_id)
                    midpoint = await clob.get_midpoint(token_id)
                    print(f"\n    Spread: {spread}")
                    print(f"    Midpoint: {midpoint}")
                except Exception as ex:
                    print(f"    Error spread/midpoint: {ex}")

        print("\n" + "-" * 60)

    await gamma.close()
    await clob.close()
    await data.close()
    print("\n[OK] Exploracion completada")


if __name__ == "__main__":
    asyncio.run(main())
