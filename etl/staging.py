# etl/staging.py
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from .utils import assign_city_slug, haversine_m, geometry_guard

MAX_CITY_DISTANCE_KM = 80  # tweak 50–120 as you prefer

def _too_far(lat, lon, city_slug, cities):
    if city_slug is None:
        return True
    city = next((c for c in cities if c.get("slug")==city_slug and c.get("center")), None)
    if not city:
        return False
    d_m = haversine_m(float(lat), float(lon), float(city["center"]["lat"]), float(city["center"]["lon"]))
    return (d_m / 1000.0) > MAX_CITY_DISTANCE_KM


def _sanitize_slug(s: str) -> str:
    return str(s).replace("/", "-").replace("\\", "-").strip()

def stage_places_raw(in_csv: str, cfg: Dict[str, Any], out_dir: str) -> Path:
    """
    Reads curated places CSV, assigns city_slug, and writes per-city Parquet:
      <out_dir>/parquet/places_raw/<slug>/data.parquet
    Keeps 'city_slug' inside each Parquet file (simple to read later).
    """
    out_root = Path(out_dir)
    pq_root = out_root / "parquet" / "places_raw"
    pq_root.mkdir(parents=True, exist_ok=True)

    # --- Load ---
    df = pd.read_csv(in_csv, low_memory=False)

    if "lat" not in df.columns or "lon" not in df.columns:
        raise SystemExit("[staging] expected columns 'lat' and 'lon' in places CSV")
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()

    # --- Assign city_slug (polygon/bbox/center + city_hint fallback) ---
    cities = cfg["cities"]

    def _city_hint(row):
        return row["city_hint"] if "city_hint" in row and pd.notna(row["city_hint"]) else None

    def _assign(row):
        try:
            return assign_city_slug(float(row["lat"]), float(row["lon"]), cities, city_hint=_city_hint(row))
        except Exception:
            return None

    def _guard_row(r):
        lat = r["lat"] if isinstance( r["lat"], float) else None; lon = r["lon"] if isinstance( r["lon"], float) else None
        if lat is not None and lon is not None and r["city_slug"]:
            return geometry_guard(lat, lon, r["city_slug"], cfg["cities"])
        return r["city_slug"]

    df["city_slug"] = df.apply(_assign, axis=1)
    df["city_slug"] = df.apply(_guard_row, axis=1)
    df = df[df["city_slug"].notna()].copy()
    df = df[~df.apply(lambda r: _too_far(r["lat"], r["lon"], r["city_slug"], cities), axis=1)].copy()
    df["city_slug"] = df["city_slug"].map(_sanitize_slug)

    if df.empty:
        raise SystemExit("[staging] no rows assigned to any city; check your geometry in config.")

    # --- Write per-city Parquet (no partition flavors) ---
    try:
        import pyarrow as pa, pyarrow.parquet as pq
    except Exception as e:
        raise SystemExit(f"[staging] PyArrow is required for Parquet writes: {e}")

    for slug, part in df.groupby("city_slug"):
        sub = pq_root / slug
        sub.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(part, preserve_index=False), sub / "data.parquet")

    # Summary (optional)
    summary = df["city_slug"].value_counts().to_dict()
    print("[staging] rows per city:", summary)

    return pq_root
