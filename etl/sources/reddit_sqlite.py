import sqlite3
import pandas as pd
from typing import Optional
from ..utils import name_similarity

def load_reddit_sqlite(db_path: str) -> Optional[pd.DataFrame]:
    if not db_path:
        return None
    conn = sqlite3.connect(db_path)
    q = '''
    SELECT
        canonical AS canonical,
        name AS name,
        city AS city,
        confidence AS confidence,
        latitude AS lat,
        longitude AS lon
    FROM reddit_pois
    '''
    df = pd.read_sql_query(q, conn)
    conn.close()
    if df.empty: return None
    out = pd.DataFrame({
        "place_id": df["canonical"].astype(str).radd("reddit:"),
        "source": "reddit",
        "name": df["name"],
        "lat": df["lat"],
        "lon": df["lon"],
        "type": None,
        "address": None,
        "city_hint": df["city"]
    })
    return out

def enrich_reddit_with_coords(df_reddit: pd.DataFrame, *coord_sources: pd.DataFrame, sim_threshold: float = 0.86) -> pd.DataFrame:
    if df_reddit is None or df_reddit.empty:
        return df_reddit
    sources = [s for s in coord_sources if s is not None and not s.empty]
    if not sources:
        return df_reddit
    base = df_reddit.copy()
    base["lat"] = base["lat"].astype("float64")
    base["lon"] = base["lon"].astype("float64")
    for s in sources:
        s2 = s[s["lat"].notna() & s["lon"].notna()].copy()
        for city, part_r in base.groupby(base["city_hint"].str.lower()):
            part_s = s2[s2["city_hint"].str.lower() == city] if "city_hint" in s2.columns else s2
            if part_s.empty: 
                continue
            sims = part_s["name"].apply(lambda x: None)
            for idx, r in part_r.iterrows():
                if pd.notna(r["lat"]) and pd.notna(r["lon"]):
                    continue
                # compute best match
                sc = part_s["name"].apply(lambda x: name_similarity(str(r["name"]), str(x)))
                j = sc.idxmax() if len(sc) else None
                if j is not None and sc[j] >= sim_threshold:
                    base.at[idx, "lat"] = part_s.at[j, "lat"]
                    base.at[idx, "lon"] = part_s.at[j, "lon"]
                    if "type" in part_s.columns:
                        base.at[idx, "type"] = part_s.at[j, "type"]
                    if "address" in part_s.columns:
                        base.at[idx, "address"] = part_s.at[j, "address"]
    return base
