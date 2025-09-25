from pathlib import Path
import argparse
import json

from etl.config_loader import load_config
from etl.airbnb.staging import stage_airbnb_dir
from etl.airbnb.reviews_jsonl import export_airbnb_reviews_jsonl

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to cities_config.generated.json")
    ap.add_argument("--airbnb-dir", required=True, help="Directory containing *_listings.csv and *_reviews.csv")
    ap.add_argument("--out-dir", default="./data", help="Output directory root")
    ap.add_argument("--places-csv", help="Path to POI Neo4j export places.csv (for NEAR linking)")
    ap.add_argument("--near-threshold-m", type=float, default=300.0)
    ap.add_argument(
        "--link-reviews-to-places",
        action="store_true",
        dest="link_reviews_to_places",
        help="Also emit review_near_place.csv by joining JSONL reviews with listing_near_place"
    )
    args = ap.parse_args()

    cfg = load_config(args.config)

    # 1) Stage Airbnb listings/hosts (CSV) as before
    listings_out, hosts_out, _reviews_out_unused = stage_airbnb_dir(args.airbnb_dir, cfg, args.out_dir)
    neo_root = Path(args.out_dir) / "exports" / "neo4j" / "airbnb"
    neo_root.mkdir(parents=True, exist_ok=True)
    print(f"Wrote Airbnb Neo4j CSVs to {neo_root}")

    # 2) Build Listing-NEAR->Place edges (same as your current code)
    if args.places_csv:
        import pandas as pd, numpy as np
        from math import radians, cos

        places = pd.read_csv(args.places_csv)
        out_rows = []

        for slug, lpart in listings_out.groupby("city_slug"):
            P = places[places["city_slug"] == slug]
            if P.empty:
                continue
            # nearest Place for each listing within threshold
            for _, row in lpart.iterrows():
                lat, lon = row["lat"], row["lon"]
                dx = (P["lon"] - lon).to_numpy() * 40075000.0 * cos(radians(lat)) / 360.0
                dy = (P["lat"] - lat).to_numpy() * 111320.0
                meters = (dx * dx + dy * dy) ** 0.5
                j = int(np.argmin(meters))
                m = float(meters[j])
                if m <= args.near_threshold_m:
                    out_rows.append({
                        "listing_id": row["listing_id"],
                        "place_id": P.iloc[j]["place_id"],
                        "meters": m,
                        "city_slug": slug
                    })

        near_df_cols = ["listing_id", "place_id", "meters", "city_slug"]
        near_path = neo_root / "listing_near_place.csv"
        if out_rows:
            import pandas as pd
            near = pd.DataFrame(out_rows, columns=near_df_cols)
            near.to_csv(near_path, index=False)
            print(f"Wrote Listing-NEAR->Place edges (rows={len(near)}) -> {near_path}")
        else:
            # still write an empty file with header so downstream is predictable
            with near_path.open("w", encoding="utf-8") as f:
                f.write(",".join(near_df_cols) + "\n")
            print("[airbnb] No listing->place edges within threshold; wrote empty file with header.")

    # 3) Export Airbnb REVIEWS as JSONL (and per-city Parquet for analysis)
    listings_csv_for_map = str(neo_root / "listings.csv")  # listing_id -> city_slug map
    emitted = export_airbnb_reviews_jsonl(
        airbnb_dir=args.airbnb_dir,
        out_dir=args.out_dir,
        listings_csv_path=listings_csv_for_map,
        jsonl_relpath="airbnb/reviews.jsonl",
        parquet_dir_relpath="airbnb_reviews",
    )
    print(f"[run_airbnb] Airbnb reviews exported to JSONL; emitted={emitted}")

    # 4) (Optional) Review-NEAR->Place by joining JSONL to listing_near_place
    if args.link_reviews_to_places:
        near_path = neo_root / "listing_near_place.csv"
        jsonl_path = neo_root / "airbnb" / "reviews.jsonl"
        r2p_path = neo_root / "review_near_place.csv"

        if not near_path.exists():
            print("[airbnb] Skipping review->place linking (listing_near_place.csv missing).")
            return
        if not jsonl_path.exists():
            print("[airbnb] Skipping review->place linking (reviews.jsonl missing).")
            return

        import pandas as pd

        # Build listing_id -> (place_id, city_slug) map from near edges
        near = pd.read_csv(near_path, dtype={"listing_id": str})
        if near.empty:
            # again, write header and exit
            with r2p_path.open("w", encoding="utf-8") as f:
                f.write("review_id,place_id,city_slug\n")
            print("[airbnb] No listing->place edges; wrote empty review_near_place.csv header.")
            return
        near_map = near.drop_duplicates(subset=["listing_id"])[["listing_id","place_id","city_slug"]]
        near_map = dict(
            (str(row.listing_id), (row.place_id, row.city_slug))
            for _, row in near_map.iterrows()
        )

        # Stream JSONL to avoid loading everything into memory
        written = 0
        header_written = False
        with open(jsonl_path, "r", encoding="utf-8") as f_in, open(r2p_path, "w", encoding="utf-8") as f_out:
            for line in f_in:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                lid = str(rec.get("listing_id") or "")
                rid = rec.get("review_id")
                if not lid or not rid:
                    continue
                hit = near_map.get(lid)
                if not hit:
                    continue
                place_id, city_slug = hit
                if not header_written:
                    f_out.write("review_id,place_id,city_slug\n")
                    header_written = True
                # review_id is already prefixed in exporter (airbnb:<id or sha1>)
                f_out.write(f"{rid},{place_id},{city_slug}\n")
                written += 1

        print(f"[airbnb] Wrote Review-NEAR->Place edges (rows={written}) -> {r2p_path}")

if __name__ == "__main__":
    main()
