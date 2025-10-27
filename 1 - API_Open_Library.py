import aiohttp
import asyncio
import pandas as pd
import logging
import os

SEARCH_URL = "https://openlibrary.org/search.json?subject=fiction&limit={limit}&page={page}"
WORK_URL = "https://openlibrary.org/works/{work_id}.json"
RATINGS_URL = "https://openlibrary.org/works/{work_id}/ratings.json"

OUTPUT_FILE = "fiction_books.csv"

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

async def fetch(session, url):
    """Hace GET y devuelve JSON (o None si falla)."""
    try:
        async with session.get(url, timeout=20) as resp:
            if resp.status == 200:
                return await resp.json()
            else:
                logging.warning(f"Error {resp.status} en {url}")
    except Exception as e:
        logging.error(f"Fallo en {url}: {e}")
    return None

async def get_work_details(session, work_id, authors):
    """Trae detalles de un work + ratings."""
    work_data = await fetch(session, WORK_URL.format(work_id=work_id))
    if not work_data:
        return None

    title = work_data.get("title")
    desc = work_data.get("description")
    if isinstance(desc, dict):
        desc = desc.get("value")

    # ratings
    ratings_data = await fetch(session, RATINGS_URL.format(work_id=work_id))
    avg_rating = None
    if ratings_data:
        avg_rating = ratings_data.get("summary", {}).get("average")

    return {
        "work_id": work_id,
        "title": title,
        "authors": ", ".join(authors) if authors else None,
        "description": desc,
        "avg_rating": avg_rating
    }

async def process_page(session, page, limit, concurrency):
    """Procesa una página completa de búsqueda."""
    search_url = SEARCH_URL.format(limit=limit, page=page)
    search_data = await fetch(session, search_url)
    if not search_data:
        return []

    tasks = []
    for doc in search_data.get("docs", []):
        work_key = doc.get("key", "")
        if not work_key.startswith("/works/"):
            continue
        work_id = work_key.split("/")[-1]
        authors = doc.get("author_name", [])
        tasks.append(get_work_details(session, work_id, authors))

    # Ejecutar con límite de concurrencia
    sem = asyncio.Semaphore(concurrency)

    async def bound_task(task):
        async with sem:
            return await task

    results = await asyncio.gather(*[bound_task(t) for t in tasks])
    return [r for r in results if r]

async def main(limit=100, pages=100, concurrency=10):
    """Orquestador principal."""
    connector = aiohttp.TCPConnector(limit_per_host=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:

        # Detectar si ya existe un CSV
        if os.path.exists(OUTPUT_FILE):
            existing = pd.read_csv(OUTPUT_FILE)
            done_ids = set(existing["work_id"].astype(str))
            logging.info(f"Reanudando. Ya hay {len(done_ids)} libros guardados.")
        else:
            done_ids = set()

        for page in range(1, pages + 1):
            logging.info(f"Procesando página {page}/{pages}...")
            results = await process_page(session, page, limit, concurrency)

            # Filtrar duplicados
            new_results = [r for r in results if r["work_id"] not in done_ids]

            if new_results:
                df = pd.DataFrame(new_results)
                # Guardar en CSV (append si existe)
                if not os.path.exists(OUTPUT_FILE):
                    df.to_csv(OUTPUT_FILE, index=False, mode="w")
                else:
                    df.to_csv(OUTPUT_FILE, index=False, mode="a", header=False)
                logging.info(f"Guardados {len(new_results)} nuevos libros.")
                done_ids.update(r["work_id"] for r in new_results)
            else:
                logging.info("No había libros nuevos en esta página.")

    logging.info("Finalizado.")

if __name__ == "__main__":
    # Traer 10,000 libros = limit=100, pages=100
    asyncio.run(main(limit=100, pages=1000, concurrency=8))