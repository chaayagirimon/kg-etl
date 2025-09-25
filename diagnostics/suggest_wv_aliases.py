# diagnostics/suggest_wv_aliases.py
from __future__ import annotations
from pathlib import Path
import argparse, json, math
import pandas as pd

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1))*math.cos(math.radians(lat2))*math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--curated-csv", default="data/curated/places_raw.csv")
    ap.add_argument("--config", required=True, help="cities_config.generated.json")
    ap.add_argument("--staged-root", default="data/parquet/places_raw")
    ap.add_argument("--near-km", type=float, default=35.0, help="radius around city center to accept WV title as alias")
    ap.add_argument("--min-count", type=int, default=8, help="only suggest aliases seen at least this many times")
    ap.add_argument("--out", default="diagnostics/wv_alias_suggestions.json")
    args = ap.parse_args()

    # 1) curated WV
    cur = pd.read_csv(args.curated_csv)
    wv  = cur[cur["source"] == "wikivoyage"].copy()

    # 2) staged assigned set to identify UNASSIGNED
    assigned = set()
    root = Path(args.staged_root)
    for sub in sorted(d for d in root.iterdir() if d.is_dir()):
        pq = sub / "data.parquet"
        if pq.exists():
            part = pd.read_parquet(pq)
        else:
            csvs = list(sub.glob("*.csv"))
            if not csvs: continue
            part = pd.concat([pd.read_csv(c) for c in csvs], ignore_index=True)
        assigned.update(part.loc[part["source"]=="wikivoyage","place_id"].astype(str))
    wv["assigned"] = wv["place_id"].astype(str).isin(assigned)
    miss = wv[~wv["assigned"]].dropna(subset=["lat","lon","city_hint"]).copy()

    # 3) centers from config
    cfg = json.load(open(args.config))
    centers = {}
    for c in cfg.get("cities", []):
        if c.get("center"):
            centers[c["slug"]] = (float(c["center"]["lat"]), float(c["center"]["lon"]))

    # 4) for each unassigned WV row, if it is within <near-km> of some city center,
    #    propose the page title (lowercased) as an alias for that city.
    proposals = {slug: {} for slug in centers}
    for _, r in miss.iterrows():
        lat, lon = float(r["lat"]), float(r["lon"])
        title = str(r["city_hint"]).strip()
        if not title: continue
        title_norm = title.lower()
        for slug, (clat, clon) in centers.items():
            if haversine_m(clat, clon, lat, lon) <= args.near_km * 1000:
                proposals[slug][title_norm] = proposals[slug].get(title_norm, 0) + 1

    # 5) keep only frequent ones (min_count)
    out = {}
    for slug, counts in proposals.items():
        keep = [t for t, n in sorted(counts.items(), key=lambda kv: kv[1], reverse=True) if n >= args.min_count]
        if keep:
            out[slug] = keep

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[suggest] wrote {args.out} (cities with suggestions: {len(out)})")

if __name__ == "__main__":
    main()
