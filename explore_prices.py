"""Explorar parámetros del endpoint prices-history."""
import asyncio
import httpx

TOKEN_ID = "100379208559626151022751801118534484742123694725746262280150222742563282755057"
BASE = "https://clob.polymarket.com"


async def try_params(interval, fidelity):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{BASE}/prices-history", params={
            "market": TOKEN_ID,
            "interval": interval,
            "fidelity": fidelity,
        })
        d = r.json()
        h = d.get("history", [])
        print(f"interval={interval}, fidelity={fidelity} -> {len(h)} puntos")
        if h:
            print(f"  Primero: {h[0]}")
            print(f"  Ultimo:  {h[-1]}")


async def main():
    # Probar diferentes combinaciones
    for interval in ["1d", "1w", "max", "all"]:
        for fidelity in [100, 500]:
            try:
                await try_params(interval, fidelity)
            except Exception as e:
                print(f"interval={interval}, fidelity={fidelity} -> ERROR: {e}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
