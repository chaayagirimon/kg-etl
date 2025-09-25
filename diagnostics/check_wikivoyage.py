# diagnostics/check_wikivoyage.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd

def _maybe_add_repo_root():
    """
    Ensure the repo root (the directory that contains 'etl/') is on sys.path.
    This avoids name-clash with any site-packages 'etl' and fixes import errors.
    """
    here = Path(__file__).resolve()
    for parent in [here.parent] + list(here.parents):
        if (parent / "etl").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            return parent
    return None

def _read_city_partition(dir_path: Path) -> pd.DataFrame:
    # Support either Parquet (preferred) or CSV fallback
    pq = dir_path / "data.parquet"
    if pq.exists():
        try:
            return pd.read_parquet(pq)
        except Exception as e:
            print(f"[warn] failed to read {pq}: {e}")
            return pd.DataFrame()
    # CSV fallback: read all csv parts in the folder
    parts = list(dir_path.glob("*.csv"))
    if parts:
        return pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    return pd.DataFrame()

def check_parser(xml_path: str, limit: int = 5):
    """
    Calls etl.wikivoyage_xml.load_wikivoyage_xml to verify rows are parsed.
    """
    try:
        from etl.sources.wikivoyage_xml import load_wikivoyage_xml
    except Exception as e:
        print(f"[error] Could not import etl.sources.wikivoyage_xml: {e}")
        print("        Make sure this script is run with your repo root on sys.path.")
        return

    df = load_wikivoyage_xml(xml_path)
    print(f"[parser] wikivoyage rows: {len(df)}")
    if df.empty:
        return
    print("[parser] by type:")
    print(df["type"].value_counts().head(10).to_string())
    print("[parser] sample rows:")
    cols = [c for c in ["place_id","name","lat","lon","type","address","city_hint"] if c in df.columns]
    print(df[cols].head(limit).to_string(index=False))

def check_curated(curated_csv: str, limit: int = 5):
    """
    Verify Wikivoyage made it into curated/places_raw.csv.
    """
    p = Path(curated_csv)
    if not p.exists():
        print(f"[curated] file not found: {p}")
        return
    df = pd.read_csv(p)
    if "source" not in df.columns:
        print(f"[curated] 'source' column missing in {p}")
        return
    wv = df[df["source"] == "wikivoyage"].copy()
    print(f"[curated] wikivoyage rows: {len(wv)} / total {len(df)}")
    if wv.empty:
        return
    if "city_hint" in wv.columns:
        print("[curated] top city_hint:")
        print(wv["city_hint"].value_counts().head(10).to_string())
    cols = [c for c in ["place_id","name","lat","lon","type","address","city_hint"] if c in wv.columns]
    print("[curated] sample rows:")
    print(wv[cols].head(limit).to_string(index=False))

def check_staged(parquet_root: str, limit: int = 10):
    """
    Verify staged per-city partitions contain Wikivoyage rows.
    Handles both Parquet ('data.parquet') and CSV fallback inside each city folder.
    """
    root = Path(parquet_root)
    if not root.exists():
        print(f"[staged] directory not found: {root}")
        return
    per_city = {}
    total = 0
    for sub in sorted([d for d in root.iterdir() if d.is_dir()]):
        df = _read_city_partition(sub)
        if df.empty or "source" not in df.columns:
            continue
        n = int((df["source"] == "wikivoyage").sum())
        if n > 0:
            per_city[sub.name] = n
            total += n
    print(f"[staged] wikivoyage rows across cities: {total}")
    if not per_city:
        return
    # print top cities with WV presence
    top = sorted(per_city.items(), key=lambda kv: kv[1], reverse=True)[:limit]
    for k, v in top:
        print(f"         {k}: {v}")

def check_assignment(parquet_root: str, limit_missing: int = 10):
    """
    Optional check: pick a few city partitions and ensure WV rows exist with city_slug.
    """
    root = Path(parquet_root)
    if not root.exists():
        return
    # If nothing per-city, skip
    subs = [d for d in root.iterdir() if d.is_dir()]
    if not subs:
        return

    # Probe up to 5 city folders
    for sub in subs[:5]:
        df = _read_city_partition(sub)
        if df.empty:
            continue
        if "source" not in df.columns:
            continue
        wv = df[df["source"] == "wikivoyage"].copy()
        if wv.empty:
            print(f"[assign] {sub.name}: 0 WV rows in staged")
            continue
        # At this point if staged-by-city exists, they already have city_slug by design.
        print(f"[assign] {sub.name}: WV rows = {len(wv)} (city_slug={sub.name})")

def main():
    _maybe_add_repo_root()

    ap = argparse.ArgumentParser(description="Diagnostics: verify Wikivoyage ingestion through parser → curated → staged.")
    ap.add_argument("--xml", help="Path to enwikivoyage-*-pages-articles.xml(.bz2) to test the parser directly")
    ap.add_argument("--curated-csv", default="data/curated/places_raw.csv", help="Path to curated places_raw.csv")
    ap.add_argument("--staged-root", default="data/parquet/places_raw", help="Root of staged per-city partitions")
    ap.add_argument("--limit", type=int, default=5, help="Rows to show in samples / city list")
    args = ap.parse_args()

    print("=== Wikivoyage Diagnostics ===")
    if args.xml:
        print("\n-- Parser check --")
        check_parser(args.xml, limit=args.limit)

    print("\n-- Curated CSV check --")
    check_curated(args.curated_csv, limit=args.limit)

    print("\n-- Staged partitions check --")
    check_staged(args.staged_root, limit=max(10, args.limit))

    print("\n-- City assignment probe --")
    check_assignment(args.staged_root, limit_missing=args.limit)

if __name__ == "__main__":
    main()
