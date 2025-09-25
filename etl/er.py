# etl/er.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple, List

import pandas as pd

# ------------------------------ Matcher (language-agnostic) ------------------------------

import re
from rapidfuzz import fuzz

_AREAL = {
    "square","park","island","bridge","market","plaza","platz","piazza",
    "boulevard","embankment","promenade"
}
_GENERIC = {
    "tower","gate","church","cathedral","synagogue","mosque","market","square","museum",
    "bridge","castle","island","statue","monument","hall","garden","park","palace","gallery","gatehouse"
}
_STOP = {
    "the","and","of","in","at","to","for","on","by","with","de","del","di","da","la","le","el","al",
    "old","new","great","little","big","upper","lower","west","east","north","south",
    "visit","explore","discover","browse","see"
}
_non_alnum = re.compile(r"[^a-z0-9]+")

def norm_name(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "-", s)
    s = re.sub(r"[^a-z0-9\-:_.,]+", "", s)
    return s[:120]

def canonical_id_for(row) -> str | None:
    name = norm_name(row["name"] or "")
    city = row.get("city_slug")
    if not name or not city:
        return None
    return f"{name}::{city}"

def _norm(s: str) -> str:
    s = (s or "").lower()
    try:
        import unicodedata
        s = unicodedata.normalize("NFKD", s)
        s = "".join(ch for ch in s if not unicodedata.combining(ch))
    except Exception:
        pass
    s = _non_alnum.sub(" ", s).strip()
    return re.sub(r"\s+", " ", s)

def _tokens(s: str) -> List[str]:
    t = [w for w in _norm(s).split() if len(w) >= 3]
    out = []
    for w in t:
        if w.endswith("ies") and len(w) > 4:
            w = w[:-3] + "y"
        elif w.endswith("es") and len(w) > 3:
            w = w[:-2]
        elif w.endswith("s") and len(w) > 3:
            w = w[:-1]
        out.append(w)
    return out

def _content_tokens(s: str) -> set:
    return set(w for w in _tokens(s) if w not in _STOP)

def _content_jaccard(a: str, b: str) -> float:
    A, B = _content_tokens(a), _content_tokens(b)
    if not A or not B:
        return 0.0
    inter = len(A & B)
    union = len(A | B)
    return inter / union if union else 0.0

def _has_substring(a: str, b: str) -> bool:
    A = " ".join(sorted(_content_tokens(a)))
    B = " ".join(sorted(_content_tokens(b)))
    x, y = (A, B) if len(A) <= len(B) else (B, A)
    return len(x) >= 3 and x in y

def _type_overlap(a: str, b: str) -> bool:
    A, B = _content_tokens(a), _content_tokens(b)
    return len((_GENERIC & A) & (_GENERIC & B)) > 0

def _is_areal(name: str) -> bool:
    toks = _content_tokens(name)
    return len(toks & _AREAL) > 0

def _name_sim(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return max(
        fuzz.token_set_ratio(a, b) / 100.0,
        fuzz.token_sort_ratio(a, b) / 100.0,
        fuzz.partial_ratio(a, b) / 100.0,
    )

def accept_pair_and_sim(r1: pd.Series, r2: pd.Series, meters: float) -> tuple[bool, float]:
    """
    Language-agnostic acceptance rule.
    r1, r2 are pandas Series with at least: name, source; optional: address.
    """
    n1 = str(r1.get("name") or "")
    n2 = str(r2.get("name") or "")
    s  = _name_sim(n1, n2)
    cj = _content_jaccard(n1, n2)
    same_source = (str(r1.get("source")) == str(r2.get("source")))
    type_hit = _type_overlap(n1, n2)
    substr   = _has_substring(n1, n2)
    areal    = _is_areal(n1) or _is_areal(n2)

    if not same_source:
        # geometry-first, then token overlap/substring
        if meters <= 65 and (s >= 0.34 or substr or cj >= 0.55):                 # e.g., “Explore Powder Tower”
            return True, s
        if meters <= 95 and (s >= 0.46 or cj >= 0.50 or (type_hit and cj >= 0.40)):
            return True, s
        # areal boost to ~130m if tokens strongly overlap
        if meters <= (130 if areal else 120) and type_hit and (cj >= 0.50 or s >= 0.52):
            return True, s
        return False, s
    else:
        # same-source: stricter (avoid merging distinct nearby venues)
        if meters <= 25 and (s >= 0.62 or substr or cj >= 0.65):
            return True, s
        if meters <= 40 and (s >= 0.72 and cj >= 0.50):
            return True, s
        return False, s

# --- legacy wrapper so existing call sites keep working (ok, score, sim) ---
_MATCHER_FLAG = {"printed": False}
def _score_pair(r1, r2, meters, _ctx=None):
    if not _MATCHER_FLAG["printed"]:
        print("[er] using new accept_pair_and_sim via _score_pair shim")
        _MATCHER_FLAG["printed"] = True
    ok, sim = accept_pair_and_sim(r1, r2, meters)
    return ok, sim, sim


# ------------------------------ ER core ------------------------------

def _meters(a: pd.Series, b: pd.Series) -> float:
    """Fast lon/lat equirectangular approximation; good under ~1-2km."""
    dy = (a["lat"] - b["lat"]) * 111_320.0
    dx = (a["lon"] - b["lon"]) * 40075_000.0 * math.cos(math.radians((a["lat"] + b["lat"]) / 2)) / 360.0
    return float((dx * dx + dy * dy) ** 0.5)

def run_er_partition(df_city: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Resolve entities within a single city partition.
    Returns:
      links: DataFrame[a,b,src_a,src_b,name_sim,meters,city_slug]
      canon: df_city + place_canonical_id
    """
    df = df_city.reset_index(drop=True).copy()
    if df.empty:
        empty_links = pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])
        return empty_links, df.assign(place_canonical_id=df.get("place_id"))

    # coarse spatial bins to prune comparisons
    df["lat_bin"] = (df["lat"] * 100).round().astype(int)
    df["lon_bin"] = (df["lon"] * 100).round().astype(int)

    pairs: List[Dict] = []
    by_bin = df.groupby(["lat_bin", "lon_bin"])
    neighbors = [(0,0),(1,0),(0,1),(-1,0),(0,-1),(1,1),(-1,-1),(1,-1),(-1,1)]
    index = {(lb, lo): idx.index.tolist() for (lb, lo), idx in by_bin}

    for (lb, lo), ids in index.items():
        cand = set(ids)
        for dlb, dlo in neighbors:
            cand |= set(index.get((lb + dlb, lo + dlo), []))
        ids_list = list(cand)
        for i in range(len(ids_list)):
            a_idx = ids_list[i]; a = df.iloc[a_idx]
            for j in range(i + 1, len(ids_list)):
                b_idx = ids_list[j]; b = df.iloc[b_idx]
                if a["source"] == b["source"] and a["place_id"] == b["place_id"]:
                    continue
                m = _meters(a, b)
                if m > 250.0:
                    continue
                ok, score, sim = _score_pair(a, b, m, {})
                if ok:
                    pairs.append({
                        "a": a["place_id"], "b": b["place_id"],
                        "src_a": a["source"], "src_b": b["source"],
                        "name_sim": sim, "meters": m,
                        "city_slug": a["city_slug"]
                    })

    # Build links DataFrame (typed, even if empty)
    if not pairs:
        links = pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])
    else:
        links = (pd.DataFrame(pairs)
                 .sort_values(["city_slug", "meters", "name_sim"], ascending=[True, True, False])
                 .reset_index(drop=True))

    # Union-Find to get canonical ids
    parent: Dict[str, str] = {}
    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent.get(x, x)
        return parent.get(x, x)

    def union(x: str, y: str):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for pid in df["place_id"].tolist():
        parent[pid] = pid
    for _, row in links.iterrows():
        union(str(row["a"]), str(row["b"]))

    root = {pid: find(pid) for pid in df["place_id"].tolist()}

    # ---- NEW: choose a canonical name per component (simple heuristic) ----
    # group rows by (city_slug, root) because ER is per city partition anyway
    df["_root"] = df["place_id"].map(root)
    comp_best_name = (
        df.assign(_clean=lambda x: x["name"].fillna("").astype(str))
          .sort_values(by=["_root", "name"], key=lambda s: s.str.len(), ascending=False)
          .groupby("_root", as_index=True)["name"]
          .first()
          .to_dict()
    )

    # Build final city-scoped canonical IDs using your existing norm_name + city_slug
    def _canon_id_for_row(row):
        best = comp_best_name.get(row["_root"], row["name"])
        name_slug = norm_name(best or row["name"] or "")
        city = row["city_slug"]
        return f"{name_slug}::{city}" if name_slug and city else None

    df_canon = df.copy()
    df_canon["place_canonical_id"] = df_canon.apply(_canon_id_for_row, axis=1)

    # Tidy
    df_canon = df_canon.drop(columns=["_root", "lat_bin", "lon_bin"], errors="ignore")

    return links, df_canon


# ------------------------------ ER driver & IO ------------------------------

def _load_places_parquet(parquet_root: str) -> pd.DataFrame:
    """
    Load places from Parquet root.
    Works with per-city layout .../<slug>/data.parquet where each file contains 'city_slug'.
    Falls back to globbing if needed.
    """
    root = Path(parquet_root)
    if not root.exists():
        raise SystemExit(f"[er] Parquet root not found: {root}")

    try:
        import pyarrow.dataset as ds
        ds_all = ds.dataset(str(root), format="parquet")  # no partition discovery; rely on files
        df = ds_all.to_table().to_pandas()
    except Exception as e:
        # robust fallback: glob files
        import glob
        import pyarrow.parquet as pq
        frames = []
        for fp in glob.glob(str(root / "*" / "*.parquet")):
            tbl = pq.read_table(fp)
            pdf = tbl.to_pandas()
            # if city_slug somehow missing in file, infer from folder name
            if "city_slug" not in pdf.columns:
                pdf["city_slug"] = Path(fp).parent.name
            frames.append(pdf)
        if not frames:
            raise SystemExit(f"[er] No Parquet files under {root} ({e})")
        df = pd.concat(frames, ignore_index=True)

    needed = {"place_id","source","name","lat","lon","city_slug"}
    missing = needed - set(df.columns)
    if missing:
        raise SystemExit(f"[er] places dataset missing columns: {sorted(missing)}")
    # clean coords
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat","lon","city_slug"]).copy()
    return df

def run_er_all(parquet_root: str, out_dir: str, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run ER for all cities under `parquet_root`, write Neo4j exports under <out_dir>/exports/neo4j:
      - place_links.csv
      - places.csv  (includes place_canonical_id)
    Returns (links_all, places_all_with_canon)
    """
    df_all = _load_places_parquet(parquet_root)

    outputs: List[Tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for slug, df_city in df_all.groupby("city_slug", dropna=True):
        links, canon = run_er_partition(df_city)
        outputs.append((slug, links, canon))

    # concat safely (avoid pandas empty concat warning)
    link_frames = [l for _, l, _ in outputs if not l.empty]
    if link_frames:
        links_all = pd.concat(link_frames, ignore_index=True)
    else:
        links_all = pd.DataFrame(columns=["a","b","src_a","src_b","name_sim","meters","city_slug"])

    canon_frames = [c for _, _, c in outputs if not c.empty]
    places_all = pd.concat(canon_frames, ignore_index=True) if canon_frames else pd.DataFrame()

    # Write Neo4j exports
    neo_root = Path(out_dir) / "exports" / "neo4j"
    neo_root.mkdir(parents=True, exist_ok=True)

    links_all.to_csv(neo_root / "place_links.csv", index=False)

    # keep a stable set of columns for places.csv; include address/type if present
    base_cols = ["place_id","source","name","lat","lon","city_slug","place_canonical_id"]
    opt_cols = [c for c in ["address","type",] if c in places_all.columns]
    places_export = places_all[base_cols + opt_cols].copy() if not places_all.empty else pd.DataFrame(columns=base_cols + ["address","type"])
    places_export.to_csv(neo_root / "places.csv", index=False)

    # NEW: write a compact mapping file the Cypher loader can use directly
    # source_place_id, canonical_id, canonical_name, city_slug
    if not places_all.empty:
        map_df = places_all[["place_id","place_canonical_id","name","city_slug"]].copy()
        map_df = map_df.rename(columns={
            "place_id": "source_place_id",
            "place_canonical_id": "canonical_id",
            "name": "canonical_name"
        })
        map_df.to_csv(neo_root / "place_canonical_map.csv", index=False)

    return links_all, places_all, len(outputs)
