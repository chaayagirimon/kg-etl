from pathlib import Path
from typing import Dict, Any, Tuple
import sqlite3, json, hashlib
import pandas as pd
import numpy as np
from etl.utils import assign_city_slug, geometry_guard, normalize_name

def _float_or_none(x):
    try:
        v = float(x)
        if not (abs(v) != float("inf")):
            return None
        return v
    except Exception:
        return None

def _city_slug_row_r(row, cfg):
    lat = _float_or_none(row.get("lat"))
    lon = _float_or_none(row.get("lon"))
    if lat is not None and lon is not None:
        slug = assign_city_slug(lat, lon, cfg["cities"])
        return geometry_guard(lat, lon, slug, cfg["cities"])
    # LAST resort: exact city name/alias match (no coords)
    cname = str(row.get("city") or "").strip().lower()
    for c in cfg["cities"]:
        names = [c["name"].lower(), c.get("slug","").lower()] + [a.lower() for a in c.get("aliases",[])]
        if cname in names:
            return c["slug"]  # keep, but will be filtered later if coords show up wrong
    return None


def _hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update((str(p) + "|").encode("utf-8"))
    return h.hexdigest()

def extract_reviews_sqlite(reviews_db: str, cfg: Dict[str, Any], out_dir: str):
    """Extract Yelp + Reddit reviews from a single SQLite DB and emit:
       - Neo4j CSV: exports/neo4j/place_reviews.csv
       - RAG JSONL: exports/rag/reviews.jsonl
    """
    out_root = Path(out_dir)
    neo = out_root / "exports" / "neo4j"
    rag = out_root / "exports" / "rag"
    neo.mkdir(parents=True, exist_ok=True)
    rag.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(reviews_db)

    # ---- Yelp ----
    q_yelp_biz = """
    SELECT
    business_id,
    business_name,
    business_alias,
    business_address,
    city,
    latitude AS lat,
    longitude AS lon
    FROM yelp_businesses
    """
    yelp_biz = pd.read_sql_query(q_yelp_biz, conn)

    # ensure a display name; fall back to alias when business_name is empty/blank
    yelp_biz["business_name"] = yelp_biz["business_name"].astype("string")
    yelp_biz["business_alias"] = yelp_biz["business_alias"].astype("string")
    yelp_biz["display_name"] = (
        yelp_biz["business_name"].str.strip().replace({"": pd.NA}).fillna(yelp_biz["business_alias"])
    )

    q_yelp_rev = """
    SELECT business_id, rating, review_text, scraped_at
    FROM yelp_business_reviews
    """
    yelp_rev = pd.read_sql_query(q_yelp_rev, conn)

    if not yelp_biz.empty and not yelp_rev.empty:
        y = yelp_rev.merge(yelp_biz, on="business_id", how="left")
        y["source"] = "yelp"
        y["place_id"] = "yelp:" + y["business_id"].astype(str)

        # city_slug from coords if possible; else by city name/aliases
        def _city_slug_row(r):
            if pd.notna(r.get("lat")) and pd.notna(r.get("lon")):
                return _city_slug_row_r(r, cfg)
            cname = str(r.get("city") or "").strip().lower()
            for c in cfg["cities"]:
                names = [c["name"].lower(), c.get("slug","").lower()] + [a.lower() for a in c.get("aliases",[])]
                if cname in names:
                    return c["slug"]
            return None

        y["city_slug"] = y.apply(_city_slug_row, axis=1)
        y["review_id"] = y.apply(
            lambda r: f"yelp:{_hash_id(r['business_id'], r.get('scraped_at',''), (r.get('review_text') or '')[:120])}",
            axis=1
        )
        y.rename(columns={"review_text":"text"}, inplace=True)

        # keep only rows with some text
        y = y[y["text"].notna() & (y["text"].astype(str).str.strip() != "")]
        yelp_out = y[[
            "review_id","source","place_id","rating","text","scraped_at",
            "city_slug","display_name","lat","lon","business_address"
        ]].copy()
    else:
        yelp_out = pd.DataFrame(columns=[
            "review_id","source","place_id","rating","text","scraped_at",
            "city_slug","display_name","lat","lon","business_address"
        ])

    # ---- Reddit ----
    q_r_pois = """
    SELECT canonical, name, city, confidence, latitude AS lat, longitude AS lon
    FROM reddit_pois
    """
    r_pois = pd.read_sql_query(q_r_pois, conn)
    q_r_reviews = """
    SELECT poi, rating, review_text, scraped_at
    FROM reddit_poi_reviews
    """
    r_revs = pd.read_sql_query(q_r_reviews, conn)
    conn.close()

    if not r_pois.empty and not r_revs.empty:
        r = r_revs.merge(r_pois, left_on="poi", right_on="canonical", how="left", suffixes=("",""))
        r["source"] = "reddit"
        r["place_id"] = "reddit:" + r["poi"].astype(str)
        # city_slug: prefer coords; else by city name match
        def _city_slug_row_r(row):
            if pd.notna(row.get("lat")) and pd.notna(row.get("lon")):
                return assign_city_slug(float(row["lat"]), float(row["lon"]), cfg["cities"])
            cname = str(row.get("city") or "").strip().lower()
            for c in cfg["cities"]:
                names = [c["name"].lower(), c.get("slug","").lower()] + [a.lower() for a in c.get("aliases",[])]
                if cname in names:
                    return c["slug"]
            return None
        r["city_slug"] = r.apply(_city_slug_row_r, axis=1)
        r["review_id"] = r.apply(lambda x: f"reddit:{_hash_id(x['poi'], x.get('scraped_at',''), (x.get('review_text') or '')[:120])}", axis=1)
        r.rename(columns={"review_text":"text", "name":"place_name"}, inplace=True)
        reddit_out = r[["review_id","source","place_id","rating","text","scraped_at","city_slug","place_name","lat","lon","city"]].copy()
    else:
        reddit_out = pd.DataFrame(columns=["review_id","source","place_id","rating","text","scraped_at","city_slug","place_name","lat","lon","city"])

    # ---- Merge & clean ----
    reviews = pd.concat([yelp_out, reddit_out], ignore_index=True)
    # Drop empties
    reviews = reviews[reviews["text"].notna() & (reviews["text"].astype(str).str.strip() != "")].copy()
    # Safe types
    if "rating" in reviews.columns:
        reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    # Write Neo4j CSV (minimal)
    neo_df = reviews[["review_id","source","place_id","rating","text","scraped_at","city_slug"]].copy()
    # neo_df.to_csv(neo / "place_reviews.csv", index=False)

    # Write RAG JSONL with richer metadata
    rag_path = rag / "reviews.jsonl"
    with rag_path.open("w", encoding="utf-8") as f:
        for _, row in reviews.iterrows():
            rec = {
                "id": row["review_id"],
                "source": row["source"],
                "place_id": row["place_id"],
                "city_slug": row.get("city_slug"),
                "rating": None if pd.isna(row.get("rating")) else float(row["rating"]),
                "text": str(row["text"]),
                "scraped_at": row.get("scraped_at"),
                "lat": None if pd.isna(row.get("lat")) else float(row["lat"]),
                "lon": None if pd.isna(row.get("lon")) else float(row["lon"]),
            }
            # Optional names/addresses if present
            if "business_name" in row and pd.notna(row["business_name"]):
                rec["place_name"] = row["business_name"]
            if "place_name" in row and pd.notna(row["place_name"]):
                rec["place_name"] = row["place_name"]
            if "business_address" in row and pd.notna(row["business_address"]):
                rec["address"] = row["business_address"]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return neo_df, rag_path
