# etl/airbnb/reviews_jsonl.py
from __future__ import annotations
from pathlib import Path
import hashlib, json, os, re, tempfile, glob
from typing import Dict, Iterable, Optional, List

import pandas as pd
import numpy as np

def _sha1_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        if p is None: p = ""
        h.update(str(p).encode("utf-8"))
        h.update(b"\x1f")
    return h.hexdigest()

def _normalize_text(s: Optional[str]) -> str:
    if s is None: return ""
    return str(s).replace("\r\n", "\n").replace("\r", "\n")

def _ensure_dirs(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _write_jsonl(rows: Iterable[dict], out_path: Path):
    _ensure_dirs(out_path)
    with out_path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def _append_parquet_per_city(df: pd.DataFrame, root: Path, city_col="city_slug"):
    if df.empty: return
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

def _sanitize_csv_to_tmp(src: Path) -> Path:
    """
    Fix the common 'space after closing quote' pattern that breaks strict CSV parsers:
      '...text..." ,...' -> '...text...",...'
    Also normalizes CRLF to LF.
    Returns a temp file path (always) so we never mutate the original.
    """
    tmp = Path(tempfile.mkstemp(prefix="abnrev_", suffix=".csv")[1])
    with src.open("r", encoding="utf-8", newline="") as f, tmp.open("w", encoding="utf-8", newline="") as out:
        for line in f:
            line = line.replace("\r", "")
            line = re.sub(r'"\s+,', '",', line)
            out.write(line)
    return tmp

def _load_listing_city_map(listings_csv: Path) -> Dict[str, str]:
    df = pd.read_csv(listings_csv, dtype={"listing_id": str})
    df = df.dropna(subset=["listing_id", "city_slug"])
    return dict(zip(df["listing_id"], df["city_slug"]))

def export_airbnb_reviews_jsonl(
    airbnb_dir: str,
    out_dir: str,
    listings_csv_path: Optional[str] = None,
    jsonl_relpath: str = "airbnb/reviews.jsonl",
    parquet_dir_relpath: str = "airbnb_reviews",
) -> int:
    """
    Build a JSONL for Neo4j ingest (APOC) and per-city Parquet for analysis.
    Includes only reviews whose listing_id exists in Airbnb listings export (so we can attach).
    - Inputs: *_reviews.csv in airbnb_dir, and listings mapping (listing_id -> city_slug)
      If listings_csv_path is None, defaults to {out_dir}/exports/neo4j/airbnb/listings.csv
    - Output:
        {out_dir}/exports/neo4j/airbnb/reviews.jsonl
        {out_dir}/parquet/airbnb_reviews/<city>.parquet
    Returns total emitted reviews.
    """
    out_root = Path(out_dir)
    neo_root = out_root / "exports" / "neo4j"
    jsonl_path = neo_root / jsonl_relpath
    pq_root = out_root / "parquet" / parquet_dir_relpath

    if jsonl_path.exists():
        jsonl_path.unlink()
    _ensure_dirs(jsonl_path)
    pq_root.mkdir(parents=True, exist_ok=True)

    # mapping listing_id -> city_slug (from your exported Airbnb listings)
    if listings_csv_path is None:
        listings_csv_path = str(neo_root / "airbnb" / "listings.csv")
    listing_city = _load_listing_city_map(Path(listings_csv_path))
    valid_ids = set(listing_city.keys())

    review_files = sorted(glob.glob(str(Path(airbnb_dir) / "*_reviews.csv")))
    if not review_files:
        print("[airbnb_reviews_jsonl] No Airbnb *_reviews.csv found.")
        return 0

    required = {"listing_id","id","date","reviewer_id","reviewer_name","comments"}
    emitted = 0
    dropped_missing_listing = 0

    for f in review_files:
        src = Path(f)
        tmp = _sanitize_csv_to_tmp(src)  # robustify quoting
        try:
            df = pd.read_csv(
                tmp,
                dtype={"listing_id": str, "id": str, "reviewer_id": str},
                low_memory=False
            )
        finally:
            try: tmp.unlink()
            except: pass

        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{src} missing required columns: {sorted(missing)}")

        df = df.rename(columns={"id": "review_id", "comments": "text"})
        df["listing_id"] = df["listing_id"].astype(str)
        df["text"] = df["text"].map(_normalize_text)
        # if review_id missing/null, deterministically synthesize one
        mask_null = df["review_id"].isna() | (df["review_id"].astype(str) == "")
        if mask_null.any():
            df.loc[mask_null, "review_id"] = [
                _sha1_id("airbnb", lid, str(dt), str(uid), str(tx)[:256])
                for lid, dt, uid, tx in df.loc[mask_null, ["listing_id","date","reviewer_id","text"]].itertuples(index=False, name=None)
            ]

        # attach city via listing mapping
        df["city_slug"] = df["listing_id"].map(listing_city)
        before = len(df)
        df = df[df["city_slug"].notna()].copy()
        dropped_missing_listing += (before - len(df))
        if df.empty:
            continue

        df["source"] = "airbnb"
        # JSONL rows
        _write_jsonl((
            dict(
                review_id=f"airbnb:{rid}",
                source="airbnb",
                listing_id=lid,   # keep for loader convenience
                text=txt,
                date=str(dt) if pd.notna(dt) else None,
                reviewer_id=str(rid2) if pd.notna(rid2) else None,
                reviewer_name=str(rname) if pd.notna(rname) else None,
                city_slug=slug
            )
            for rid, lid, txt, dt, rid2, rname, slug in df[["review_id","listing_id","text","date","reviewer_id","reviewer_name","city_slug"]].itertuples(index=False, name=None)
        ), jsonl_path)

        # Parquet per city
        _append_parquet_per_city(
            df[["review_id","listing_id","text","date","reviewer_id","reviewer_name","city_slug"]],
            pq_root
        )
        emitted += len(df)

    print(f"[airbnb_reviews_jsonl] wrote JSONL → {jsonl_path} ; total emitted = {emitted}")
    if dropped_missing_listing:
        print(f"[airbnb_reviews_jsonl] dropped {dropped_missing_listing} rows with listing_id not in listings.csv (no city to attach).")
    print(f"[airbnb_reviews_jsonl] per-city Parquet dir → {pq_root}")
    return emitted
