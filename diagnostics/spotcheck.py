# diagnostics/spotcheck.py
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Tuple, Dict, List

import numpy as np
import pandas as pd

# Reuse your name normalizer/similarity if available
try:
    from etl.utils import name_similarity
except Exception:
    # very small fallback if utils isn't importable (less robust)
    import re, unicodedata
    def _norm(s: str) -> str:
        s = unicodedata.normalize("NFKD", s or "").encode("ascii","ignore").decode("ascii")
        s = re.sub(r"[^a-z0-9 ]+", " ", s.lower())
        return re.sub(r"\s+", " ", s).strip()
    def name_similarity(a: str, b: str) -> float:
        a, b = _norm(a), _norm(b)
        if not a or not b:
            return 0.0
        A, B = set(a.split()), set(b.split())
        if not A or not B:
            return 0.0
        return len(A & B) / len(A | B)


# ------------------------- IO helpers ------------------------- #

def enrich_pairs_with_names(pairs: pd.DataFrame, places_csv_path: str) -> pd.DataFrame:
    """Add name/address/coords for A and B from exports/neo4j/places.csv."""
    if pairs.empty:
        return pairs
    places = pd.read_csv(Path(places_csv_path), low_memory=False)
    keep = places[["place_id","name","address","lat","lon","source","city_slug"]].copy()
    a = pairs.merge(keep.add_prefix("a_"), left_on="a", right_on="a_place_id", how="left")
    b = a.merge(keep.add_prefix("b_"), left_on="b", right_on="b_place_id", how="left")
    cols = [
        "city_slug","a","a_name","a_address","a_source","a_lat","a_lon",
        "b","b_name","b_address","b_source","b_lat","b_lon","name_sim","meters"
    ]
    return b[cols]


def load_places_df(parquet_root: str, city: str | None = None) -> pd.DataFrame:
    """
    Robust Parquet loader that works with:
      A) Hive partitions:  .../places_raw/city_slug=<slug>/part-*.parquet
      B) Directory layout: .../places_raw/<slug>/part-*.parquet
      C) Per-city files:   .../places_raw/<slug>/data.parquet  (city_slug kept in file)
    """
    root = Path(parquet_root)

    # Preferred: pyarrow.dataset
    try:
        import pyarrow as pa
        import pyarrow.dataset as ds

        try:
            # Try HIVE discovery first
            ds_hive = ds.dataset(root, format="parquet", partitioning="hive")
            t = ds_hive.to_table()
            df = t.to_pandas()
        except Exception:
            # Try DIRECTORY flavor (needs a partition schema)
            part_schema = pa.schema([pa.field("city_slug", pa.string())])
            ds_dir = ds.dataset(root, format="parquet",
                                partitioning=ds.partitioning(part_schema, flavor="directory"))
            t = ds_dir.to_table()
            df = t.to_pandas()
    except Exception:
        # Fallback: walk per-city subfolders (C)
        import glob
        import pyarrow.parquet as pq
        frames = []
        for fp in glob.glob(str(root / "*" / "*.parquet")):
            tbl = pq.read_table(fp)
            pdf = tbl.to_pandas()
            if "city_slug" not in pdf.columns:
                # infer from folder name
                pdf["city_slug"] = Path(fp).parent.name
            frames.append(pdf)
        if not frames:
            raise SystemExit(f"[spotcheck] No Parquet files found under {root}")
        df = pd.concat(frames, ignore_index=True)

    # Standardize columns we rely on
    needed = {"place_id", "source", "name", "lat", "lon", "city_slug"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"[spotcheck] places dataset missing columns: {sorted(missing)}")
    # Ensure numeric coords
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    if city:
        want = {s.strip().lower() for s in city.split(",")}
        df = df[df["city_slug"].str.lower().isin(want)].copy()

    return df


def load_links_csv(exports_root: str) -> pd.DataFrame:
    path = Path(exports_root) / "place_links.csv"
    if not path.exists():
        # Empty placeholder if links not exported (or for cities with no links)
        return pd.DataFrame(columns=["a", "b", "src_a", "src_b", "name_sim", "meters", "city_slug"])
    df = pd.read_csv(path)
    # standardize column presence
    for c in ["city_slug", "name_sim", "meters", "src_a", "src_b"]:
        if c not in df.columns:
            df[c] = np.nan
    return df


def load_places_csv(exports_root: str) -> pd.DataFrame | None:
    path = Path(exports_root) / "places.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, low_memory=False)
    # columns often present: place_id, source, name, lat, lon, type, address, city_slug, place_canonical_id
    return df


# ------------------------- Pair scanning ------------------------- #

def _approx_meters(lat1, lon1, lat2, lon2) -> float:
    # equirectangular approximation is fine under ~1-2km
    # dx: meters along longitude
    dx = (lon2 - lon1) * 40075000.0 * np.cos(np.radians((lat1 + lat2) * 0.5)) / 360.0
    # dy: meters along latitude
    dy = (lat2 - lat1) * 111320.0
    return float(np.hypot(dx, dy))


def _bin_key(lat: float, lon: float, size_deg_lat=0.0015, size_deg_lon=0.0020) -> Tuple[int, int]:
    # ~0.001 deg lat ≈ 111m; use a slightly larger tile for safety.
    return (int(np.floor(lat / size_deg_lat)), int(np.floor(lon / size_deg_lon)))


def generate_candidate_pairs(dfc: pd.DataFrame,
                             max_scan_per_bin: int = 400):
    """
    Yield position pairs (i, j) using local 0..N-1 indices.
    dfc MUST be a per-city DataFrame with a contiguous index (we reset it upstream).
    """
    # Build bin -> list[pos] mapping using positions, not original indices
    bins: Dict[Tuple[int, int], List[int]] = {}
    latv = dfc["lat"].to_numpy()
    lonv = dfc["lon"].to_numpy()

    for pos in range(len(dfc)):
        key = _bin_key(float(latv[pos]), float(lonv[pos]))
        bins.setdefault(key, []).append(pos)

    neighbor_offsets = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1)]
    seen = set()

    for (bi, bj), idxs in bins.items():
        # cap per-bin to avoid quadratic blowups
        if len(idxs) > max_scan_per_bin:
            idxs = idxs[:max_scan_per_bin]

        for di, dj in neighbor_offsets:
            nbr = (bi + di, bj + dj)
            if nbr not in bins:
                continue
            targets = bins[nbr]
            if len(targets) > max_scan_per_bin:
                targets = targets[:max_scan_per_bin]

            for i in idxs:
                for j in targets:
                    if i >= j:  # upper triangle only
                        continue
                    key = (i, j)
                    if key in seen:
                        continue
                    seen.add(key)
                    yield i, j


def build_unlinked_samples(df_city: pd.DataFrame,
                           links_set: set[Tuple[str, str]],
                           borderline_min: float = 100.0,
                           borderline_max: float = 120.0) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return:
      - cross_borderline: cross-source pairs 100–120 m, NOT linked
      - cross_riskiest:   cross-source ≤120 m & low name_sim (≤0.30), NOT linked
      - same_suspicious:  same-source ≤120 m & name_sim ≥0.50 (potential dupes)
    Safe on empty/one-row cities and on non-contiguous indices.
    """
    # Use local 0..N-1 positions to avoid IndexError
    dfc = df_city.reset_index(drop=True)
    n = len(dfc)
    if n < 2:
        cols = ["a", "b", "src_a", "src_b", "name_sim", "meters", "city_slug"]
        empty = pd.DataFrame(columns=cols)
        return empty.copy(), empty.copy(), empty.copy()

    # Local arrays
    lat = dfc["lat"].to_numpy()
    lon = dfc["lon"].to_numpy()
    name = dfc["name"].astype(str).to_numpy()
    pid  = dfc["place_id"].astype(str).to_numpy()
    src  = dfc["source"].astype(str).to_numpy()
    city = str(dfc["city_slug"].iloc[0]) if "city_slug" in dfc.columns else ""

    rows_borderline, rows_risky, rows_same = [], [], []

    for i, j in generate_candidate_pairs(dfc):
        d = _approx_meters(lat[i], lon[i], lat[j], lon[j])
        if d > 120.0:
            continue

        a, b = pid[i], pid[j]
        sa, sb = src[i], src[j]
        if (a, b) in links_set or (b, a) in links_set:
            continue

        sim = name_similarity(name[i], name[j])

        if sa != sb:
            if borderline_min <= d <= borderline_max:
                rows_borderline.append((a, b, sa, sb, sim, d, city))
            if d <= 120.0 and sim <= 0.30:
                rows_risky.append((a, b, sa, sb, sim, d, city))
        else:
            if d <= 120.0 and sim >= 0.50:
                rows_same.append((a, b, sa, sb, sim, d, city))

    cols = ["a", "b", "src_a", "src_b", "name_sim", "meters", "city_slug"]
    cross_borderline = pd.DataFrame(rows_borderline, columns=cols)
    cross_riskiest   = pd.DataFrame(rows_risky,     columns=cols)
    same_suspicious  = pd.DataFrame(rows_same,      columns=cols)

    # Pretty order
    if not cross_borderline.empty:
        cross_borderline = cross_borderline.sort_values(["city_slug", "meters", "name_sim"],
                                                        ascending=[True, True, False]).reset_index(drop=True)
    if not cross_riskiest.empty:
        cross_riskiest = cross_riskiest.sort_values(["city_slug", "name_sim", "meters"],
                                                    ascending=[True, True, True]).reset_index(drop=True)
    if not same_suspicious.empty:
        same_suspicious = same_suspicious.sort_values(["city_slug", "meters", "name_sim"],
                                                      ascending=[True, True, False]).reset_index(drop=True)

    return cross_borderline, cross_riskiest, same_suspicious


# ------------------------- Main ------------------------- #

def main():
    ap = argparse.ArgumentParser(description="ER spotcheck & risk sampling")
    ap.add_argument("--parquet-root", required=True, help="Path to data/parquet/places_raw")
    ap.add_argument("--exports", required=True, help="Path to data/exports/neo4j")
    ap.add_argument("--city", help="Comma-separated list of city slugs to check (default: all)")
    ap.add_argument("--out-dir", default="./diagnostics", help="Where to write CSVs")
    ap.add_argument("--borderline-min", type=float, default=100.0)
    ap.add_argument("--borderline-max", type=float, default=120.0)
    ap.add_argument("--max-scan-per-bin", type=int, default=400,
                    help="Cap comparisons per spatial bin to avoid O(n^2) blowups")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_places_df(args.parquet_root, city=args.city)
    links = load_links_csv(args.exports)
    places_csv = load_places_csv(args.exports)

    # Build a set of linked pairs (undirected)
    link_pairs = set()
    if not links.empty:
        # ensure string type
        L = links[["a", "b"]].dropna().astype(str).values.tolist()
        for a, b in L:
            link_pairs.add((a, b))
            link_pairs.add((b, a))

    # Summary block
    print(f"[spotcheck] places rows = {len(df):,}")
    if places_csv is not None and "place_canonical_id" in places_csv.columns:
        # cluster stats from canonical ids
        ccounts = places_csv.groupby("city_slug")["place_canonical_id"].nunique().sum()
        total_places = len(places_csv)
        singletons = (places_csv.groupby("place_canonical_id").size() == 1).sum()
        print(f"[spotcheck] exported places.csv rows = {total_places:,} | canonical clusters = {ccounts:,} | singletons = {singletons:,}")
    print(f"[spotcheck] links rows = {len(links):,}")

    # Per-city scanning
    all_borderline = []
    all_risky = []
    all_same = []

    for slug, df_city in df.groupby("city_slug"):
        if df_city.empty:
            continue
        cb, cr, ss = build_unlinked_samples(
            df_city,
            links_set=link_pairs,
            borderline_min=args.borderline_min,
            borderline_max=args.borderline_max,
        )
        all_borderline.append(cb)
        all_risky.append(cr)
        all_same.append(ss)

        # quick city summary
        src_counts = df_city["source"].value_counts().to_dict()
        print(f"[spotcheck][{slug}] places={len(df_city):3d} by source={src_counts} "
              f"| borderline={len(cb):3d} risky={len(cr):3d} same_suspicious={len(ss):3d}")

    cross_borderline = pd.concat(all_borderline, ignore_index=True) if any(len(x) for x in all_borderline) else pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])
    cross_riskiest   = pd.concat(all_risky,     ignore_index=True) if any(len(x) for x in all_risky)     else pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])
    same_suspicious  = pd.concat(all_same,      ignore_index=True) if any(len(x) for x in all_same)      else pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])
    # --- Suggest safe promotions using the ER matcher (if available) ---
    try:
        from etl.er import accept_pair_and_sim
    except Exception:
        accept_pair_and_sim = None

    def _promote(df_pairs: pd.DataFrame, exports_root: str) -> pd.DataFrame:
        if accept_pair_and_sim is None or df_pairs.empty:
            return pd.DataFrame(columns=["a","b","city_slug","name_sim","meters"])
        # bring in names/sources to evaluate the rule
        import pandas as pd
        places = pd.read_csv(Path(exports_root) / "places.csv", low_memory=False)
        keep = places[["place_id","name","source","city_slug"]].copy()
        a = df_pairs.merge(keep.add_prefix("a_"), left_on="a", right_on="a_place_id", how="left")
        b = a.merge(keep.add_prefix("b_"), left_on="b", right_on="b_place_id", how="left")

        rows = []
        for _, r in b.iterrows():
            ok, sim = accept_pair_and_sim(
                pd.Series({"name": r.get("a_name"), "source": r.get("a_source")}),
                pd.Series({"name": r.get("b_name"), "source": r.get("b_source")}),
                float(r.get("meters", 1e9))
            )
            if ok:
                rows.append({"a": r["a"], "b": r["b"], "city_slug": r["city_slug"], "name_sim": sim, "meters": r["meters"]})
        return pd.DataFrame(rows)

    promote = pd.concat([
        _promote(cross_borderline, args.exports),
        _promote(cross_riskiest,   args.exports)
    ], ignore_index=True).drop_duplicates()

    if not promote.empty:
        promote_path = out_dir / "spot_promote_suggested.csv"
        promote.to_csv(promote_path, index=False)
        print(f"[spotcheck] Suggested safe promotions: {len(promote)} -> {promote_path}")
    else:
        print("[spotcheck] Suggested safe promotions: 0")

    # Write outputs (same filenames you saw earlier)
    p1 = out_dir / "spot_cross_borderline_100_120m.csv"
    p2 = out_dir / "spot_cross_riskiest.csv"
    p3 = out_dir / "spot_same_weak_far.csv"   # keeping original filename for continuity
    p4 = out_dir / "spot_same_low_sim.csv"    # extra: very low sim same-source near pairs

    # Split same-source into "weak_far" (>=50m & sim>=0.5) and "low_sim" (<0.25)
    if not same_suspicious.empty:
        same_weak_far = same_suspicious[same_suspicious["meters"] >= 50.0].copy()
        same_low_sim = same_suspicious[same_suspicious["name_sim"] < 0.25].copy()
    else:
        same_weak_far = same_suspicious.copy()
        same_low_sim  = same_suspicious.copy()

    places_csv_path = str(Path(args.exports) / "places.csv")
    cross_borderline = enrich_pairs_with_names(cross_borderline, places_csv_path)
    cross_riskiest   = enrich_pairs_with_names(cross_riskiest,   places_csv_path)
    same_weak_far    = enrich_pairs_with_names(same_weak_far,    places_csv_path)
    same_low_sim     = enrich_pairs_with_names(same_low_sim,     places_csv_path)

    cross_borderline.to_csv(p1, index=False)
    cross_riskiest.to_csv(p2, index=False)
    same_weak_far.to_csv(p3, index=False)
    same_low_sim.to_csv(p4, index=False)

    print(f"[spotcheck] Wrote:\n - {p1}\n - {p2}\n - {p3}\n - {p4}")


if __name__ == "__main__":
    main()
