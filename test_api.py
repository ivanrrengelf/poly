import asyncio
import json
from src.data.gamma_client import GammaClient
from src.data.clob_client import ClobClient

async def test():
    g = GammaClient()
    c = ClobClient()
    ev = await g.get_macro_finance_events(max_pages=1)
    m = ev[0]['markets'][0]
    token = json.loads(m['clobTokenIds'])[0]
    cond = m['conditionId']
    
    print(f"Token: {token}")
    print(f"Cond: {cond}")
    
    try:
        book1 = await c.get_book(token)
        print("Book with token worked")
    except Exception as e:
        print(f"Book with token failed: {e}")
        
    try:
        book2 = await c.get_book(cond)
        print("Book with cond worked")
    except Exception as e:
        print(f"Book with cond failed: {e}")
        
    await g.close()
    await c.close()

if __name__ == "__main__":
    asyncio.run(test())
