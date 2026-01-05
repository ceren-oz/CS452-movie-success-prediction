import aiohttp
import asyncio
import pandas as pd
import os
from tqdm import tqdm

API_KEY = os.getenv("TMDB_API_KEY")
if not API_KEY:
    raise ValueError("TMDB_API_KEY environment variable not set")

BASE_URL = "https://api.themoviedb.org/3"

MAX_CONCURRENT_REQUESTS = 20
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# ---------------- API ----------------
async def fetch_json(session, url, params):
    async with semaphore:
        async with session.get(url, params=params, timeout=20) as response:
            response.raise_for_status()
            return await response.json()

async def discover_movies(session, year, page):
    return await fetch_json(
        session,
        f"{BASE_URL}/discover/movie",
        {
            "api_key": API_KEY,
            "primary_release_year": year,
            "page": page,
            "sort_by": "popularity.desc",
        },
    )

async def get_movie_full(session, movie_id):
    return await fetch_json(
        session,
        f"{BASE_URL}/movie/{movie_id}",
        {
            "api_key": API_KEY,
            "append_to_response": "credits",
        },
    )

# ---------------- YEAR PROCESS ----------------
async def process_year(session, year, movies_data):
    first_page = await discover_movies(session, year, 1)
    total_pages = min(first_page.get("total_pages", 1), 500)

    page_bar = tqdm(
        total=total_pages,
        desc=f"{year} pages",
        leave=False
    )

    for page in range(1, total_pages + 1):
        data = await discover_movies(session, year, page)
        movies = data.get("results", [])

        tasks = [get_movie_full(session, m["id"]) for m in movies]

        for coro in asyncio.as_completed(tasks):
            try:
                data = await coro
            except Exception:
                continue

            if data.get("revenue", 0) <= 0 or data.get("vote_average", 0) <= 0:
                continue

            credits = data.get("credits", {})

            movies_data.append({
                "budget": data.get("budget", 0),
                "runtime": data.get("runtime", 0),
                "genres": [g["name"] for g in data.get("genres", [])],
                "production_companies": [c["name"] for c in data.get("production_companies", [])],
                "production_countries": [c["iso_3166_1"] for c in data.get("production_countries", [])],
                "release_year": year,
                "popularity": data.get("popularity", 0),
                "vote_count": data.get("vote_count", 0),
                "original_language": data.get("original_language", ""),
                "cast_count": len(credits.get("cast", [])),
                "crew_count": len(credits.get("crew", [])),
                "revenue": data.get("revenue", 0),
                "rating": data.get("vote_average", 0),
            })

        page_bar.update(1)

    page_bar.close()

# ---------------- MAIN ----------------
async def main():
    movies_data = []

    timeout = aiohttp.ClientTimeout(total=None)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for year in tqdm(range(2000, 2025), desc="Years"):
            await process_year(session, year, movies_data)

    df = pd.DataFrame(movies_data)
    df.to_csv("tmdb_movies_2000_2024_async.csv", index=False)
    print("Final shape:", df.shape)

if __name__ == "__main__":
    asyncio.run(main())
