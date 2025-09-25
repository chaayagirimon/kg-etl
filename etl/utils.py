# etl/utils.py
import re, unicodedata
from math import radians, cos, sin, asin, sqrt
from typing import Optional, List, Iterable, Tuple, Dict, Any
from rapidfuzz import fuzz
# utils.py (add these)
import unicodedata, re

# --- add near the top ---
import unicodedata, re

def _deaccent_lower(s: str) -> str:
    return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii").lower().strip()

def wv_title_candidates(title: str):
    """
    Generate robust match keys for a Wikivoyage page title.
    Examples:
      'Paris/1st arrondissement' -> ['paris/1st arrondissement','paris']
      'Porto (Portugal)'         -> ['porto (portugal)','porto']
      'Rome/Trevi'               -> ['rome/trevi','rome']
    """
    t0 = _deaccent_lower(title or "")
    cands = [t0]
    # root before slash
    if "/" in t0:
        cands.append(t0.split("/", 1)[0].strip())
    # drop trailing parenthetical qualifier
    cands.append(re.sub(r"\s*\([^)]*\)\s*$", "", t0).strip())
    # soft remove generic suffixes (rarely needed, harmless if no match)
    cands.append(re.sub(r"\b(district|province|region|prefecture|county)$", "", t0).strip())
    # dedupe / prune empties
    out = []
    for c in cands:
        if c and c not in out:
            out.append(c)
    return out

# --- replace your existing _hint_match_city_slug with this version ---
def _hint_match_city_slug(city_hint: Optional[str], cities: list) -> Optional[str]:
    if not city_hint:
        return None
    cands = wv_title_candidates(city_hint)
    for c in cities:
        names = [c.get("name",""), c.get("slug","")] + list(c.get("aliases", []))
        norm = {_deaccent_lower(x) for x in names if x}
        for q in cands:
            if q in norm:
                return c.get("slug")
    return None

# etl/utils.py
USE_NAME_SYNONYMS = False  # default OFF for multi-city

def clean_name(s: str, *, apply_synonyms: bool = USE_NAME_SYNONYMS) -> str:
    # ... your current folding/lowercasing ...
    if apply_synonyms:
        # apply TOKEN_SYNONYMS/PHRASE_SYNONYMS here
        pass
    return s

def haversine_m(lat1, lon1, lat2, lon2):
    R = 6371000.0
    dlat, dlon = radians(lat2-lat1), radians(lon2-lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
    return 2 * R * asin(sqrt(a))

def normalize_name(s: Optional[str]) -> str:
    if not s:
        return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"str\.|strasse\b", "strasse", s)
    s = re.sub(r"\bpl\.|platz\b", "platz", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def name_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    a, b = normalize_name(a), normalize_name(b)
    return max(
        fuzz.token_set_ratio(a, b)/100.0,
        fuzz.QRatio(a, b)/100.0,
        fuzz.token_sort_ratio(a, b)/100.0
    )

def street_tokens(addr: Optional[str]) -> List[str]:
    if not addr: return []
    s = normalize_name(addr)
    toks = [t for t in s.split() if len(t) >= 3]
    return toks

def jaccard(a: List[str], b: List[str]) -> float:
    if not a or not b: return 0.0
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

# ---- NEW: geometry helpers ----
def _bbox_contains(lat: float, lon: float, bbox) -> bool:
    """
    bbox can be [min_lat, min_lon, max_lat, max_lon]
    or {'min_lat':..,'min_lon':..,'max_lat':..,'max_lon':..}
    """
    if bbox is None: return False
    try:
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            min_lat, min_lon, max_lat, max_lon = bbox
        else:
            min_lat, min_lon = bbox["min_lat"], bbox["min_lon"]
            max_lat, max_lon = bbox["max_lat"], bbox["max_lon"]
        return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)
    except Exception:
        return False

def _as_latlon_pairs(poly) -> List[Tuple[float, float]]:
    """
    Accepts:
      - [(lat, lon), (lat, lon), ...]
      - [{'lat':..,'lon':..}, ...]
    Returns list of (lat, lon)
    """
    pts = []
    for p in poly or []:
        if isinstance(p, (list, tuple)) and len(p) >= 2:
            pts.append((float(p[0]), float(p[1])))
        elif isinstance(p, dict) and "lat" in p and "lon" in p:
            pts.append((float(p["lat"]), float(p["lon"])))
    return pts

def _point_in_polygon(lat: float, lon: float, poly) -> bool:
    """Ray-casting algorithm. Polygon expected as list of (lat, lon)."""
    pts = _as_latlon_pairs(poly)
    if len(pts) < 3: 
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        yi, xi = pts[i][0], pts[i][1]
        yj, xj = pts[j][0], pts[j][1]
        # Edge crosses the horizontal ray?
        intersect = ((xi > lon) != (xj > lon)) and (
            lat < (yj - yi) * (lon - xi) / ((xj - xi) + 1e-12) + yi
        )
        if intersect:
            inside = not inside
        j = i
    return inside

def geometry_guard(lat, lon, slug, cities):
    """Return slug if (lat,lon) ∈ slug's bbox; else None."""
    if lat is None or lon is None or slug is None:
        return None
    by_slug = {c["slug"]: c for c in cities}
    c = by_slug.get(slug)
    if not c:
        return None
    return slug if _bbox_contains(float(lat), float(lon), c.get("bbox")) else None

def assign_city_slug(lat: Optional[float], lon: Optional[float],
                     cities: List[Dict[str, Any]],
                     city_hint: Optional[str] = None) -> Optional[str]:
    """
    Priority:
      1) polygon contains point
      2) bbox contains point
      3) nearest center within radius_km
      4) fallback: hint ONLY IF coordinates are missing
    """
    has_coords = (lat is not None) and (lon is not None)

    if has_coords:
        # 1) polygons
        for c in cities:
            poly = c.get("polygon")
            if poly and _point_in_polygon(lat, lon, poly):
                return c.get("slug")

        # 2) bboxes
        for c in cities:
            bbox = c.get("bbox")
            if bbox and _bbox_contains(lat, lon, bbox):
                return c.get("slug")

        # 3) nearest center within radius
        best_slug, best_m = None, float("inf")
        for c in cities:
            center = c.get("center")
            if not center:
                continue
            try:
                d = haversine_m(lat, lon, float(center["lat"]), float(center["lon"]))
            except Exception:
                continue
            if d <= c.get("radius_km", 25) * 1000 and d < best_m:
                best_slug, best_m = c.get("slug"), d
        if best_slug:
            return best_slug

        # Coords exist but didn't match any city → do NOT use hint
        return None

    # No coords → you may use the hint
    return _hint_match_city_slug(city_hint, cities)

# === BEGIN: bbox helpers (append to etl/utils.py) ============================

from pathlib import Path
import json
from typing import Tuple, Dict, Any
import math
import pandas as pd

def load_cities_config(path: str | Path) -> Dict[str, Any]:
    """
    Load cities_config.generated.json
    Returns dict with key 'cities' (list of city dicts).
    """
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def index_cities_by_slug(cfg: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Build slug->city_config index.
    """
    by_slug: Dict[str, Dict[str, Any]] = {}
    for c in cfg.get("cities", []):
        slug = c.get("slug")
        if slug:
            by_slug[slug] = c
    return by_slug

def _deg_buffer_at_lat_km(buffer_km: float, lat_deg: float) -> Tuple[float, float]:
    """
    Convert km buffer to degrees (lat_deg, lon_deg) at a given latitude.
    1 deg lat ~ 111.32 km; 1 deg lon ~ 111.32 * cos(lat) km.
    """
    if buffer_km <= 0:
        return 0.0, 0.0
    lat_deg_per_km = 1.0 / 111.32
    lon_deg_per_km = 1.0 / (111.32 * max(0.01, math.cos(math.radians(lat_deg))))
    return buffer_km * lat_deg_per_km, buffer_km * lon_deg_per_km

def expand_bbox(bbox: list[float], buffer_km: float, at_lat: float | None = None) -> Tuple[float, float, float, float]:
    """
    Expand bbox [min_lat, min_lon, max_lat, max_lon] by buffer_km.
    If at_lat is None, uses the bbox center latitude.
    """
    min_lat, min_lon, max_lat, max_lon = bbox
    if at_lat is None:
        at_lat = 0.5 * (min_lat + max_lat)
    dlat, dlon = _deg_buffer_at_lat_km(buffer_km, at_lat)
    return (min_lat - dlat, min_lon - dlon, max_lat + dlat, max_lon + dlon)

def point_in_bbox(lat: float, lon: float, bbox: list[float], buffer_km: float = 0.0) -> bool:
    """
    True if (lat, lon) is inside bbox (optionally expanded by buffer_km).
    bbox format: [min_lat, min_lon, max_lat, max_lon]
    """
    if lat is None or lon is None:
        return False
    min_lat, min_lon, max_lat, max_lon = expand_bbox(bbox, buffer_km, at_lat=lat)
    return (min_lat <= lat <= max_lat) and (min_lon <= lon <= max_lon)

def filter_df_in_city_bbox(
    df: pd.DataFrame,
    cities_index: Dict[str, Dict[str, Any]],
    city_col: str = "city_slug",
    lat_col: str = "lat",
    lon_col: str = "lon",
    buffer_km: float = 0.0,
) -> pd.DataFrame:
    """
    Keep only rows whose (lat, lon) fall inside the bbox for row[city_col].
    Rows with missing city/lat/lon or missing bbox are dropped.
    """
    if df is None or df.empty:
        return df

    data = df.copy()
    mask = pd.Series(False, index=data.index)

    for slug, city_cfg in cities_index.items():
        bbox = city_cfg.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        idx = (data[city_col] == slug)
        if not idx.any():
            continue

        sub = data.loc[idx, [lat_col, lon_col]].astype(float)
        lat_vals = sub[lat_col]
        lon_vals = sub[lon_col]

        # buffer in degrees (lat uses mean; lon adjusts per-row via cos(lat))
        dlat, _ = _deg_buffer_at_lat_km(buffer_km, lat_vals.mean() if len(lat_vals) else 0.0)
        lon_deg_per_km = 1.0 / (111.32 * (lat_vals.apply(lambda v: max(0.01, math.cos(math.radians(v))))))
        dlon_series = lon_deg_per_km * buffer_km

        min_lat, min_lon, max_lat, max_lon = bbox
        inside = (
            (lat_vals >= (min_lat - dlat)) &
            (lat_vals <= (max_lat + dlat)) &
            (lon_vals >= (min_lon - dlon_series)) &
            (lon_vals <= (max_lon + dlon_series))
        )
        mask.loc[idx] = inside.fillna(False)

    return data.loc[mask].copy()

# === END: bbox helpers =======================================================
