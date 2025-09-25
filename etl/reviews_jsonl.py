# etl/reviews_jsonl.py
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import os
import sqlite3
from typing import Dict, Iterable, Optional
import re
import pandas as pd
import numpy as np

# --- helpers / fallbacks ------------------------------------------------------

# slugify (kept for any non-critical fallbacks; not used for city fallback anymore)
try:
    from .utils import slugify
except Exception:
    import unicodedata
    def slugify(text: str) -> str:
        s = unicodedata.normalize("NFKD", str(text)).encode("ascii","ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
        s = re.sub(r"-{2,}", "-", s)
        return s

# assign_city_slug does polygon → bbox → radius matching
from .utils import assign_city_slug

# load city config (bbox/polygons/centers)
try:
    from .config_loader import load_config  # expects cities_config.generated.json shape
except Exception:
    def load_config(path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

# --- local utils --------------------------------------------------------------

def _sha1_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None:
            p = ""
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")  # separator
    return h.hexdigest()

def _normalize_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    # normalize CRLF/CR to LF; Neo4j stores as-is
    return str(s).replace("\r\n", "\n").replace("\r", "\n")

def _ensure_dirs(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _append_parquet_per_city(df: pd.DataFrame, root: Path, city_col="city_slug"):
    """
    Robust append that avoids pyarrow partition flavors that tripped earlier.
    Writes one Parquet per city under data/parquet/place_reviews/<slug>.parquet
    """
    if df.empty:
        return
    _ensure_dirs(root / "dummy")
    for slug, part in df.groupby(city_col):
        if not isinstance(slug, str) or not slug:
            continue
        out_pq = root / f"{slug}.parquet"
        if out_pq.exists():
            prev = pd.read_parquet(out_pq)
            pd.concat([prev, part], ignore_index=True).to_parquet(out_pq, index=False)
        else:
            part.to_parquet(out_pq, index=False)

def _yield_sql_chunks(conn: sqlite3.Connection, sql: str, chunksize: int = 100_000):
    for chunk in pd.read_sql_query(sql, conn, chunksize=chunksize):
        yield chunk

def _write_jsonl(rows: Iterable[dict], out_path: Path):
    _ensure_dirs(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

# --- place index + strict geometry guard -------------------------------------

def _load_place_index(places_csv: Path) -> Dict[str, dict]:
    """
    Return mapping: place_id -> {"city_slug": str, "lat": float|None, "lon": float|None}
    We use this to attach city_slugs and to geometry-check using place coordinates.
    Tolerant to schema variations.
    """
    try:
        df = pd.read_csv(places_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to read {places_csv}: {e}")

    cols = {c.lower(): c for c in df.columns}
    need_place = cols.get("place_id")
    # 'city_slug' is the canonical; some older exports might have 'city'
    need_city  = cols.get("city_slug") or cols.get("city")
    lat_col    = cols.get("lat") or cols.get("latitude")
    lon_col    = cols.get("lon") or cols.get("longitude")

    usecols = [c for c in [need_place, need_city, lat_col, lon_col] if c]
    if not usecols or (need_place is None or need_city is None):
        raise RuntimeError(
            f"{places_csv} must include at least ['place_id','city_slug'] (or 'city'); "
            f"found columns: {list(df.columns)}"
        )

    df = df[usecols].copy()
    rename = {}
    if need_place: rename[need_place] = "place_id"
    if need_city:  rename[need_city]  = "city_slug"
    if lat_col:    rename[lat_col]    = "lat"
    if lon_col:    rename[lon_col]    = "lon"
    df = df.rename(columns=rename)

    df = df.dropna(subset=["place_id", "city_slug"])
    if "lat" not in df.columns: df["lat"] = np.nan
    if "lon" not in df.columns: df["lon"] = np.nan

    out: Dict[str, dict] = {}
    for _, r in df.iterrows():
        pid = str(r["place_id"])
        out[pid] = {
            "city_slug": str(r["city_slug"]),
            "lat": None if pd.isna(r["lat"]) else float(r["lat"]),
            "lon": None if pd.isna(r["lon"]) else float(r["lon"]),
        }
    return out

def _safe_city_from_place_meta(meta: Optional[dict], cities: list) -> Optional[str]:
    """
    Strict guard:
      - If (lat,lon) exist → recompute via assign_city_slug (polygon→bbox→radius).
        DROP (return None) if recomputed != hinted city (cross-city leak).
      - If no coords → keep the hinted city (no better signal available here).
    """
    if not meta:
        return None
    hint = meta.get("city_slug")
    lat  = meta.get("lat")
    lon  = meta.get("lon")
    if lat is not None and lon is not None:
        fixed = assign_city_slug(float(lat), float(lon), cities, city_hint=hint)
        return fixed if fixed == hint else None  # strict drop on mismatch
    return hint

# --- main export --------------------------------------------------------------

def export_place_reviews_jsonl(
    reviews_db_path: str,
    places_csv_path: str,
    out_dir: str,
    jsonl_name: str = "place_reviews.jsonl",
    parquet_dir_name: str = "place_reviews",
    cities_config_path: str = "cities_config.generated.json",  # NEW: needed for geometry checks
):
    """
    Build a single JSONL for Neo4j ingest (via APOC) and a per-city Parquet store for analysis.
    Includes only reviews whose place_id exists in places.csv.
    - Yelp reviews -> place_id = 'yelp:<business_id>'
    - Reddit reviews -> place_id = 'reddit:<poi_id>' (from reddit_pois.canonical)
    - Wikivoyage -> top-k distinct snippets per place (pseudo-reviews)

    Output:
      data/exports/neo4j/place_reviews.jsonl
      data/parquet/place_reviews/<city_slug>.parquet
    """
    out_root = Path(out_dir)
    neo_root = out_root / "exports" / "neo4j"
    pq_root  = out_root / "parquet" / parquet_dir_name
    jsonl_path = neo_root / jsonl_name

    # Fresh file
    if jsonl_path.exists():
        jsonl_path.unlink()
    _ensure_dirs(jsonl_path)
    pq_root.mkdir(parents=True, exist_ok=True)

    # Load place index (+ coords) and valid ids
    place_index = _load_place_index(Path(places_csv_path))
    valid_ids = set(place_index.keys())

    # Load city config (polygons/bboxes/centers)
    cfg = load_config(cities_config_path)
    cities = cfg["cities"]

    # Connect SQLite
    conn = sqlite3.connect(reviews_db_path)
    conn.row_factory = sqlite3.Row

    emitted = 0

    # --- Yelp (chunked) -------------------------------------------------------
    sql_yelp = """
        SELECT b.business_id, r.rating, r.review_text AS text, r.scraped_at
        FROM yelp_business_reviews r
        JOIN yelp_businesses b ON b.business_id = r.business_id
    """
    for chunk in _yield_sql_chunks(conn, sql_yelp):
        # Build place_id and filter to known places
        chunk["place_id"] = "yelp:" + chunk["business_id"].astype(str)
        chunk = chunk[chunk["place_id"].isin(valid_ids)].copy()
        if chunk.empty:
            continue

        # Normalize fields
        chunk["source"] = "yelp"
        chunk["text"]   = chunk["text"].map(_normalize_text)
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk["review_id"] = [
            f"yelp:{_sha1_id(pid, ts, str(t)[:512])}"
            for pid, ts, t in zip(chunk["place_id"], chunk["scraped_at"], chunk["text"])
        ]

        # Strict geometry guard per row using place_index meta
        meta_series = chunk["place_id"].map(place_index)  # dicts or NaN
        chunk["city_slug"] = [ _safe_city_from_place_meta(m, cities) for m in meta_series ]
        # DROP out-of-geometry or unknown
        chunk = chunk.dropna(subset=["city_slug"])
        if chunk.empty:
            continue

        # JSONL
        _write_jsonl(
            (dict(
                review_id=row.review_id,
                source=row.source,
                place_id=row.place_id,
                rating=None if pd.isna(row.rating) else float(row.rating),
                text=row.text,
                scraped_at=row.scraped_at,
                city_slug=row.city_slug
            ) for _, row in chunk.iterrows()),
            jsonl_path
        )
        # Parquet (analysis)
        _append_parquet_per_city(
            chunk[["review_id","source","place_id","rating","text","scraped_at","city_slug"]],
            pq_root
        )
        emitted += len(chunk)

    # --- Reddit (chunked) -----------------------------------------------------
    # reddit_poi_reviews.poi references reddit_pois.canonical; we use canonical as poi_key
    sql_reddit = """
        SELECT
            p.canonical AS poi_key,  -- canonical key
            p.city,                  -- (kept for debugging; no fallback use)
            r.rating, r.review_text AS text, r.scraped_at
        FROM reddit_poi_reviews r
        JOIN reddit_pois p ON r.poi = p.canonical
    """
    for chunk in _yield_sql_chunks(conn, sql_reddit):
        chunk["place_id"] = "reddit:" + chunk["poi_key"].astype(str)
        chunk = chunk[chunk["place_id"].isin(valid_ids)].copy()
        if chunk.empty:
            continue

        chunk["source"] = "reddit"
        chunk["text"]   = chunk["text"].map(_normalize_text)
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk["review_id"] = [
            f"reddit:{_sha1_id(pid, ts, str(t)[:512])}"
            for pid, ts, t in zip(chunk["place_id"], chunk["scraped_at"], chunk["text"])
        ]

        # Strict geometry guard (NO text-city fallback)
        meta_series = chunk["place_id"].map(place_index)
        chunk["city_slug"] = [ _safe_city_from_place_meta(m, cities) for m in meta_series ]
        chunk = chunk.dropna(subset=["city_slug"])
        if chunk.empty:
            continue

        # JSONL
        _write_jsonl(
            (dict(
                review_id=row.review_id,
                source=row.source,
                place_id=row.place_id,
                rating=None if pd.isna(row.rating) else float(row.rating),
                text=row.text,
                scraped_at=row.scraped_at,
                city_slug=row.city_slug
            ) for _, row in chunk.iterrows()),
            jsonl_path
        )
        # Parquet (analysis)
        _append_parquet_per_city(
            chunk[["review_id","source","place_id","rating","text","scraped_at","city_slug"]],
            pq_root
        )
        emitted += len(chunk)

    # --- Wikivoyage "pseudo-reviews" (top-k distinct per place) ---------------
    # (Kept intact, but guarded the city the same way for consistency)
    from rapidfuzz import fuzz

    WIKI_LINK_PIPE = re.compile(r'\[\[([^|\]]+)\|([^\]]+)\]\]')  # [[Title|Alt]]
    WIKI_LINK      = re.compile(r'\[\[([^\]]+)\]\]')             # [[Title]]
    TEMPLATES      = re.compile(r'\{\{[^}]+\}\}')                # {{...}}
    WS             = re.compile(r'\s+')

    def clean_wv_text(t: str) -> str:
        if not isinstance(t, str):
            return t
        t = WIKI_LINK_PIPE.sub(r'\2', t)
        t = WIKI_LINK.sub(r'\1', t)
        t = TEMPLATES.sub('', t)
        t = WS.sub(' ', t).strip()
        return t

    def top_k_distinct(texts, k=2, sim_thresh=90):
        """Return up to k texts that are not near-duplicates (longest-first)."""
        texts = [clean_wv_text(t) for t in texts if isinstance(t, str) and t.strip()]
        kept = []
        for t in sorted(texts, key=len, reverse=True):
            if all(fuzz.token_set_ratio(t, u) < sim_thresh for u in kept):
                kept.append(t)
            if len(kept) >= k:
                break
        return kept

    # Read places and select WV rows
    wv_df = pd.read_csv(places_csv_path)
    if "source" in wv_df.columns:
        wv_df = wv_df[wv_df["source"] == "wikivoyage"].copy()
    else:
        wv_df = wv_df.iloc[0:0].copy()  # no WV rows present

    if not wv_df.empty:
        # Choose text: prefer 'desc', fallback to name+address
        fallback_text = (wv_df.get("name", pd.Series("", index=wv_df.index)).fillna("") +
                         " – " +
                         wv_df.get("address", pd.Series("", index=wv_df.index)).fillna("")).str.strip(" –")
        wv_df["text"] = wv_df["desc"] if "desc" in wv_df.columns else None
        wv_df["text"] = wv_df["text"].map(clean_wv_text).str.slice(0, 1200)
        wv_df["text"] = wv_df["text"].fillna(fallback_text)

        # Keep only rows with non-empty text and valid place ids
        wv_df = wv_df[wv_df["text"].notna() & (wv_df["text"].str.strip() != "")]
        wv_df = wv_df[wv_df["place_id"].isin(valid_ids)].copy()

    if not wv_df.empty:
        # Group texts per place_id and pick top-k distinct
        grouped = (
            wv_df.groupby("place_id")["text"]
            .apply(list)
            .reset_index(name="texts")
        )

        k = 2
        sim_thresh = 90
        grouped["kept_texts"] = grouped["texts"].apply(lambda lst: top_k_distinct(lst, k=k, sim_thresh=sim_thresh))

        # Explode back to rows with strict city guard
        rows = []
        for pid, texts in zip(grouped["place_id"], grouped["kept_texts"]):
            meta = place_index.get(pid)
            safe_city = _safe_city_from_place_meta(meta, cities)
            if not safe_city:
                continue
            for idx, t in enumerate(texts):
                txt = t[:1200]  # trim for RAG
                rid = f"wikivoyage:{_sha1_id(pid, txt)}"
                rows.append({
                    "review_id": rid,
                    "source": "wikivoyage",
                    "place_id": pid,
                    "rating": None,
                    "text": txt,
                    "scraped_at": None,
                    "city_slug": safe_city
                })

        if rows:
            _write_jsonl((row for row in rows), jsonl_path)
            df_out = pd.DataFrame(rows, columns=["review_id","source","place_id","rating","text","scraped_at","city_slug"])
            _append_parquet_per_city(df_out, pq_root)
            emitted += len(rows)

    conn.close()
    print(f"[reviews_jsonl] wrote JSONL → {jsonl_path} ; total reviews emitted = {emitted}")
    print(f"[reviews_jsonl] per-city Parquet dir → {pq_root}")
