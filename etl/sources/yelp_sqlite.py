import sqlite3
import pandas as pd
from typing import Optional

def load_yelp_sqlite(db_path: str) -> Optional[pd.DataFrame]:
    if not db_path: 
        return None
    conn = sqlite3.connect(db_path)
    q = '''
    SELECT
        business_id AS yelp_id,
        business_name AS name,
        business_address AS address,
        city,
        latitude AS lat,
        longitude AS lon
    FROM yelp_businesses
    WHERE latitude IS NOT NULL AND longitude IS NOT NULL AND name IS NOT NULL
    '''
    df = pd.read_sql_query(q, conn)
    conn.close()
    if df.empty: return None
    out = pd.DataFrame({
        "place_id": df["yelp_id"].astype(str).radd("yelp:"),
        "source": "yelp",
        "name": df["name"],
        "lat": df["lat"],
        "lon": df["lon"],
        "type": "restaurant",  # fallback without categories table
        "address": df["address"],
        "city_hint": df["city"]
    })
    return out
