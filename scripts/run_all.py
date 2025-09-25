# scripts/run_all.py
"""
Multi-city ETL runner

Pipeline steps (in order):
  1) (optional) Build curated/places_raw.csv    [--build-places]
  2) Stage to Parquet partitions                [default unless --er-only]
  3) Run ER + export Neo4j CSVs                [default unless --stage-only]
  4) Export reviews JSONL/Parquet (Yelp/Reddit/WV) AFTER ER if --reviews-db is given

Key rules:
  - --stage-only and --er-only are mutually exclusive.
  - The reviews export depends on exports/neo4j/places.csv (created by ER), so it
    runs *after* ER by default. If you pass --stage-only, the reviews export will
    only run if places.csv already exists from a previous ER; otherwise it’s skipped
    with a clear message.
"""

from pathlib import Path
import argparse
import sys as _sys
import os as _os

# Capture original argv BEFORE any import might mutate sys.argv
_ORIG_ARGV = _sys.argv[1:].copy()

from etl.config_loader import load_config
from etl.sources.unify_sqlite import build_places_raw_sqlite
from etl.staging import stage_places_raw
from etl.er import run_er_all
from etl.reviews_jsonl import export_place_reviews_jsonl


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="run_all", description="Multi-city ETL runner")

    ap.add_argument("--config", required=True,
                    help="Path to cities_config.generated.json")
    ap.add_argument("--sources-dir", default="./data/raw",
                    help="Directory with raw inputs (SQLite, XML, etc.)")
    ap.add_argument("--out-dir", default="./data",
                    help="Output directory root (curated/, parquet/, exports/)")

    # Inputs to build curated/places_raw.csv
    ap.add_argument("--build-places", action="store_true",
                    help="Build curated/places_raw.csv from reviews.db and/or Wikivoyage XML")
    ap.add_argument("--reviews-db",
                    help="Path to SQLite reviews.db (Yelp+Reddit). If provided, reviews JSONL export will run AFTER ER.")
    ap.add_argument("--wv-xml",
                    help="Path to Wikivoyage XML dump (e.g., ./data/raw/enwikivoyage-YYYYMMDD-pages-articles.xml)")

    # Stage/ER mode switches (mutually exclusive)
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument("--stage-only", action="store_true",
                    help="Run staging only (no ER).")
    mx.add_argument("--er-only", action="store_true",
                    help="Run ER only (requires existing parquet/places_raw).")

    # Misc
    ap.add_argument("--chunksize", type=int, default=100_000,
                    help="Chunk size for SQL reads (if used by downstream fns).")
    ap.add_argument("--verbose", action="store_true",
                    help="Print extra debug information.")
    return ap


def _eprint(msg: str):
    print(msg, file=_sys.stderr)


def main(argv=None):
    if argv is None:
        argv = list(_ORIG_ARGV)

    ap = _build_parser()
    args, unknown = ap.parse_known_args(argv)
    if unknown:
        _eprint(f"[run_all] warning: ignoring unknown args: {unknown}")
    if args.verbose or _os.getenv("RUNALL_DEBUG") == "1":
        print(f"[run_all] args = {args}")

    # Resolve paths
    out_dir = Path(args.out_dir)
    curated_csv = out_dir / "curated" / "places_raw.csv"
    parquet_root = out_dir / "parquet" / "places_raw"
    exports_neo4j = out_dir / "exports" / "neo4j"
    places_csv = exports_neo4j / "places.csv"

    # Load config early to fail fast if missing
    cfg = load_config(args.config)
    if args.verbose:
        print(f"[run_all] loaded config with {len(cfg.get('cities', []))} cities from {args.config}")

    # Decide which steps to run
    do_build = bool(args.build_places)
    do_stage = not args.er_only
    do_er = not args.stage_only
    do_reviews_export = bool(args.reviews_db)

    # ---------------------------------------------------------------------
    # Step 1: build curated/places_raw.csv (optional)
    # ---------------------------------------------------------------------
    if do_build:
        curated_csv.parent.mkdir(parents=True, exist_ok=True)
        if args.verbose:
            print(f"[run_all] building places_raw.csv → {curated_csv}")
        build_places_raw_sqlite(
            args.sources_dir,
            str(curated_csv),
            args.reviews_db,
            args.wv_xml
        )
    else:
        if not curated_csv.exists() and do_stage:
            raise SystemExit(
                f"[run_all] Missing {curated_csv}. "
                f"Run with --build-places or provide the file."
            )

    # ---------------------------------------------------------------------
    # Step 2: staging (unless --er-only)
    # ---------------------------------------------------------------------
    if do_stage:
        if args.verbose:
            print(f"[run_all] staging from {curated_csv} → {out_dir}")
        pq_root = stage_places_raw(str(curated_csv), cfg, str(out_dir))
        print(f"[run_all] Staged to Parquet at: {pq_root}")
    else:
        if args.verbose:
            print("[run_all] --er-only set; skipping staging.")

    # ---------------------------------------------------------------------
    # Step 3: ER + exports (unless --stage-only)
    # ---------------------------------------------------------------------
    if do_er:
        if not parquet_root.exists():
            raise SystemExit(
                f"[run_all] Missing {parquet_root}. "
                f"Run staging first (omit --er-only), or ensure the directory exists."
            )
        if args.verbose:
            print(f"[run_all] running ER over {parquet_root}")
        links_df, places_df, n_cities = run_er_all(str(parquet_root), str(out_dir), cfg)
        print(f"[run_all] ER complete for {n_cities} cities. Neo4j exports → {exports_neo4j}")
    else:
        if args.verbose:
            print("[run_all] --stage-only set; skipping ER.")

    # ---------------------------------------------------------------------
    # Step 4: Reviews export (Yelp/Reddit/WV) — requires places.csv
    #         We run AFTER ER by default so places.csv is fresh.
    # ---------------------------------------------------------------------
    if do_reviews_export:
        if not places_csv.exists():
            msg = (f"[run_all] WARNING: {places_csv} not found. "
                   f"Reviews export requires places.csv (created by ER).")
            if args.stage_only:
                # In stage-only mode, quietly skip reviews export if places.csv doesn't exist.
                _eprint(msg + " Skipping reviews export due to --stage-only.")
            else:
                # If not stage-only, this means ER didn't produce it → treat as fatal
                raise SystemExit(msg + " Run ER (omit --stage-only) first.")
        else:
            if args.verbose:
                print(f"[run_all] exporting reviews JSONL using {args.reviews_db} and {places_csv}")
            export_place_reviews_jsonl(
                reviews_db_path=args.reviews_db,
                places_csv_path=str(places_csv),
                out_dir=str(out_dir),
                jsonl_name="place_reviews.jsonl",
                parquet_dir_name="place_reviews",
                cities_config_path=args.config,  # IMPORTANT: pass config for geometry guard
            )
    else:
        if args.verbose:
            print("[run_all] --reviews-db not provided; skipping reviews JSONL export.")

    print("[run_all] DONE.")


if __name__ == "__main__":
    # Do NOT read sys.argv directly—some imports may mutate it
    main(_ORIG_ARGV)
