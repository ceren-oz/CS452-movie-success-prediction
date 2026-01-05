import requests
import pandas as pd
import time
import os
from tqdm import tqdm
from datetime import datetime

API_KEY = os.getenv("TMDB_API_KEY")

if not API_KEY:
    raise ValueError("TMDB_API_KEY environment variable not set")

BASE_URL = "https://api.themoviedb.org/3"

# Get movies by year
def discover_movies(year, page=1):
    url = f"{BASE_URL}/discover/movie"
    params = {
        "api_key": API_KEY,
        "primary_release_year": year,
        "page": page,
        "sort_by": "popularity.desc"
    }
    return requests.get(url, params=params).json()

# Get movie details: Budget, Revenue, Runtime, Companies, Countries
def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY}
    return requests.get(url, params=params).json()

# Get credits (cast & crew count)
def get_movie_credits(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}/credits"
    params = {"api_key": API_KEY}
    return requests.get(url, params=params).json()

# Data collection from 2000 to 2025
movies_data = []

for year in range(2000, 2025):
    print(f"Collecting movies for year {year}")

    try:
        first_page = discover_movies(year, page=1)
        total_pages = min(first_page.get("total_pages", 1), 5)
    except Exception as e:
        print(f"Skipping year {year}: {e}")
        continue

    for page in range(1, total_pages + 1):
        results = discover_movies(year, page)["results"]

        for movie in tqdm(results, leave=False):
            movie_id = movie["id"]

            try:
                details = get_movie_details(movie_id)
                credits = get_movie_credits(movie_id)
            except Exception:
                continue

            # Filter: must have revenue & rating
            if details.get("revenue", 0) <= 0 or details.get("vote_average", 0) <= 0:
                continue

            movies_data.append({
                # -------- INPUT FEATURES (X) --------
                "budget": details.get("budget", 0),
                "runtime": details.get("runtime", 0),
                "genres": [g["name"] for g in details.get("genres", [])],
                "production_companies": [c["name"] for c in details.get("production_companies", [])],
                "production_countries": [c["iso_3166_1"] for c in details.get("production_countries", [])],
                "release_year": year,
                "popularity": details.get("popularity", 0),
                "vote_count": details.get("vote_count", 0),
                "original_language": details.get("original_language", ""),
                "cast_count": len(credits.get("cast", [])),
                "crew_count": len(credits.get("crew", [])),

                # -------- TARGETS (Y) --------
                "revenue": details.get("revenue", 0),
                "rating": details.get("vote_average", 0)
            })

            time.sleep(0.2)  # rate-limit protection

# Create DataFrame
df = pd.DataFrame(movies_data)
print("Dataset shape: ", df.shape)


# Create classification target based on revenue (flop/average/hit)
# the revenue range is split into three equal-sized groups
# Two boundaries:
low_thresh = df["revenue"].quantile(0.33) # 33% of movies earn less than or equal to this revenue
high_thresh = df["revenue"].quantile(0.66) # 66% of movies earn less than or equal to this revenue

def revenue_class(revenue):
    if revenue <= low_thresh:
        return "Flop"
    elif revenue <= high_thresh:
        return "Average"
    else:
        return "Hit"

df["success_class"] = df["revenue"].apply(revenue_class)

print(df["success_class"].value_counts())

df.to_csv("tmdb_movies_2000_2024.csv", index=False)

