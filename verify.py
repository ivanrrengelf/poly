"""Script de verificación rápida de los clientes API."""
import asyncio
import json
from src.data.gamma_client import GammaClient
from src.data.clob_client import ClobClient


async def main():
    gamma = GammaClient()
    clob = ClobClient()

    # 1. Obtener algunos eventos activos
    print("=" * 60)
    print("GAMMA API — Eventos activos")
    print("=" * 60)
    events = await gamma.get_events(limit=5)
    print(f"Recibidos: {len(events)} eventos\n")

    for e in events[:5]:
        title = e.get("title", "?")
        tags = e.get("tags", [])
        markets = e.get("markets", [])
        print(f"  > {title}")
        print(f"     Tags: {tags}")
        print(f"     Mercados: {len(markets)}")

        # 2. Para el primer mercado, obtener precio del CLOB
        if markets:
            market = markets[0]
            token_ids = market.get("clobTokenIds")
            if token_ids:
                # Parsear si viene como string JSON
                if isinstance(token_ids, str):
                    token_ids = json.loads(token_ids)
                if token_ids:
                    token_id = token_ids[0]
                    try:
                        price = await clob.get_price(token_id)
                        print(f"     Precio (CLOB): {price}")
                    except Exception as ex:
                        print(f"     Error CLOB: {ex}")
        print()

    # 3. Tags disponibles
    print("=" * 60)
    print("GAMMA API — Tags disponibles")
    print("=" * 60)
    tags = await gamma.get_tags()
    if isinstance(tags, list):
        for t in tags[:20]:
            if isinstance(t, dict):
                print(f"  [{t.get('label', t.get('slug', t))}]")
            else:
                print(f"  [{t}]")

    await gamma.close()
    await clob.close()
    print("\n[OK] Verificacion completada")


if __name__ == "__main__":
    asyncio.run(main())
