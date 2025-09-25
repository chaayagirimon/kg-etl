# etl/airbnb/staging.py
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import glob
import re

from ..utils import assign_city_slug


LISTINGS_REQUIRED = ("id", "name", "latitude", "longitude", "host_id", "price")
REVIEWS_REQUIRED = ("listing_id", "id", "date", "reviewer_id", "reviewer_name", "comments")


def _read_listings(files: List[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        df = pd.read_csv(
            f, low_memory=False,
            dtype={  # force string types for identifiers
                "id": "string",
                "host_id": "string",
            }
        )
        missing = [c for c in LISTINGS_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _read_reviews(files: List[str]) -> pd.DataFrame:
    parts = []
    for f in files:
        df = pd.read_csv(
            f, low_memory=False,
            dtype={  # force string types for identifiers
                "listing_id": "string",
                "id": "string",
                "reviewer_id": "string",
            }
        )
        missing = [c for c in REVIEWS_REQUIRED if c not in df.columns]
        if missing:
            raise ValueError(f"{f} missing columns: {missing}")
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

def _parse_price(x):
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "")
    m = re.search(r"([0-9]+(\.[0-9]+)?)", s)
    return float(m.group(1)) if m else np.nan

_id_tail = re.compile(r"\.0+$")

def _canon_id_str(s) -> str | None:
    if pd.isna(s):
        return None
    s = str(s).strip()
    # common CSV issues: numeric read as float → "123.0", or whitespace
    s = _id_tail.sub("", s)
    return s or None

def stage_airbnb_dir(airbnb_dir: str, cfg: Dict[str, Any], out_dir: str):
    """
    Ingest Airbnb listings/reviews CSVs and write:
      - Parquet (per city):   <out_dir>/airbnb/parquet/listings/<city_slug>/data.parquet
      - Parquet (reviews):    <out_dir>/airbnb/parquet/reviews/data.parquet
      - Neo4j CSV exports:    <out_dir>/exports/neo4j/airbnb/{listings,hosts,reviews,listing_city}.csv

    Notes:
      * No Hive/Directory partitioning is used → no 'city_slug=<slug>' folders.
      * Each city Parquet keeps the 'city_slug' column inside the file for simple reading later.
    """
    out_root = Path(out_dir)
    pq_root = out_root / "airbnb" / "parquet"
    neo_root = out_root / "exports" / "neo4j" / "airbnb"
    pq_root.mkdir(parents=True, exist_ok=True)
    neo_root.mkdir(parents=True, exist_ok=True)

    # Accept both "<city>_listings.csv" & "listings_<city>.csv"
    # (Use your preferred pattern if you want to restrict)
    listing_files = sorted(
        glob.glob(str(Path(airbnb_dir) / "*_listings.csv")) +
        glob.glob(str(Path(airbnb_dir) / "listings_*.csv"))
    )
    review_files = sorted(
        glob.glob(str(Path(airbnb_dir) / "*_reviews.csv")) +
        glob.glob(str(Path(airbnb_dir) / "reviews_*.csv"))
    )

    listings = _read_listings(listing_files)
    reviews = _read_reviews(review_files)

    listings["id"] = listings["id"].map(_canon_id_str)
    listings["host_id"] = listings.get("host_id").map(_canon_id_str) if "host_id" in listings.columns else listings.get("host_id")

    if not reviews.empty:
        reviews["listing_id"] = reviews["listing_id"].map(_canon_id_str)
        reviews["id"] = reviews["id"].map(_canon_id_str)               # review_id column
        if "reviewer_id" in reviews.columns:
            reviews["reviewer_id"] = reviews["reviewer_id"].map(_canon_id_str)

    if listings.empty:
        raise SystemExit("No Airbnb listings CSVs found (expected *_listings.csv or listings_*.csv).")

    # Robust price parsing
    listings["price"] = listings["price"].apply(_parse_price)

    # City assignment (geometry-based, falls back as defined in utils.assign_city_slug)
    def _assign(row):
        try:
            return assign_city_slug(float(row["latitude"]), float(row["longitude"]), cfg["cities"])
        except Exception:
            return None

    listings["city_slug"] = listings.apply(_assign, axis=1)
    listings = listings[listings["city_slug"].notna()].copy()

    # ---- Write Parquet (no partitioning; keep city_slug in the file) ----
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as e:
        raise SystemExit(f"[airbnb] PyArrow is required for Parquet writes: {e}")

    # Listings: one Parquet per city, e.g., .../airbnb/parquet/listings/amsterdam/data.parquet
    for slug, part in listings.groupby("city_slug"):
        sub = pq_root / "listings" / slug
        sub.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(part), sub / "data.parquet")

    # Reviews: join to listings to get city_slug, then write per-city Parquets
    if not reviews.empty:
        # map listing_id -> city_slug
        id2city = (
            listings[["id", "city_slug"]]
            .dropna()
            .assign(id=lambda d: d["id"].astype(str))
            .set_index("id")["city_slug"]
            .to_dict()
        )

        reviews = reviews.copy()
        reviews["listing_id"] = reviews["listing_id"].astype(str)
        reviews["city_slug"] = reviews["listing_id"].map(id2city)

        # keep only reviews whose listing we ingested and assigned to a city
        missing = reviews["city_slug"].isna().sum()
        if missing:
            print(f"[airbnb] dropping {missing} review rows whose listing_id was not found in listings/city mapping")
        reviews = reviews[reviews["city_slug"].notna()].copy()

        # write one Parquet per city: .../airbnb/parquet/reviews/<slug>/data.parquet
        for slug, part in reviews.groupby("city_slug"):
            sub = pq_root / "reviews" / slug
            sub.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pandas(part), sub / "data.parquet")


    # ---- Neo4j CSV exports ----
    listings_out = listings[[
        "id", "name", "latitude", "longitude", "room_type",
        "accommodates", "bedrooms", "bathrooms", "price", "host_id", "city_slug"
    ]].copy()
    listings_out.rename(columns={"id": "listing_id", "latitude": "lat", "longitude": "lon"}, inplace=True)

    # Hosts (derive from listings; fill missing columns if absent)
    host_cols = ["host_id", "host_name", "host_is_superhost", "host_listings_count", "host_total_listings_count"]
    for c in host_cols:
        if c not in listings.columns:
            listings[c] = np.nan
    hosts_out = listings[host_cols].drop_duplicates(subset=["host_id"])

    # Reviews (rename to stable schema)
    if not reviews.empty:
        reviews_out = reviews.rename(columns={"id": "review_id", "comments": "text"})[
            ["review_id", "listing_id", "date", "text", "reviewer_id", "reviewer_name"]
        ]
    else:
        reviews_out = pd.DataFrame(columns=["review_id", "listing_id", "date", "text", "reviewer_id", "reviewer_name"])

    # Write CSV exports for Neo4j loaders
    listings_out.to_csv(neo_root / "listings.csv", index=False)
    hosts_out.to_csv(neo_root / "hosts.csv", index=False)
    reviews_out.to_csv(neo_root / "reviews.csv", index=False)
    listings_out[["listing_id", "city_slug"]].drop_duplicates().to_csv(neo_root / "listing_city.csv", index=False)

    return listings_out, hosts_out, reviews_out
