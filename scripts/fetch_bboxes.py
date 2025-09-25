# scripts/fetch_bboxes.py
import os, time, json, sys, pathlib, requests, argparse

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = os.environ.get(
    "NOMINATIM_UA",
    "SustainableTourismETL/1.0 (contact: youremail@example.com)"
)
CACHE_PATH = pathlib.Path("./data/cache/nominatim_bboxes.json")


def load_json(p):
    p = pathlib.Path(p)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(p, obj):
    p = pathlib.Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def fetch_bbox(q: str):
    # Nominatim boundingbox is [south, north, west, east] (strings)
    params = {"q": q, "format": "json", "limit": 1, "addressdetails": 1}
    r = requests.get(NOMINATIM_URL, params=params, headers={"User-Agent": USER_AGENT}, timeout=20)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    d = data[0]
    bb = d.get("boundingbox")
    if not bb or len(bb) != 4:
        return None
    south, north, west, east = map(float, bb)
    bbox = [south, west, north, east]  # [min_lat, min_lon, max_lat, max_lon]
    lat = float(d.get("lat"))
    lon = float(d.get("lon"))
    return {"bbox": bbox, "center": {"lat": lat, "lon": lon}}


def best_queries(city: dict):
    name = city.get("name") or city.get("slug") or ""
    country = city.get("country")
    aliases = city.get("aliases", [])
    base = [name] + aliases
    if country:
        for s in base:
            if s:
                yield f"{s}, {country}"
    for s in base:
        if s:
            yield s
    if name:
        yield f"{name}, Europe"


def _invalid_center(c):
    return (not isinstance(c, dict)) or ("lat" not in c) or ("lon" not in c)


def _invalid_bbox(b):
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return False
    if isinstance(b, dict):
        return not all(k in b for k in ("min_lat", "min_lon", "max_lat", "max_lon"))
    return True


def _normalize_bbox(b):
    # Accept either list [min_lat, min_lon, max_lat, max_lon] or dict with keys
    if isinstance(b, (list, tuple)) and len(b) == 4:
        return [float(b[0]), float(b[1]), float(b[2]), float(b[3])]
    return [float(b["min_lat"]), float(b["min_lon"]), float(b["max_lat"]), float(b["max_lon"])]


def upsert_geo(city: dict, hit: dict, overwrite: bool = False) -> bool:
    """Write bbox/center into city; return True if anything changed."""
    changed = False

    # center
    cur_center = city.get("center")
    if overwrite or (cur_center is None) or _invalid_center(cur_center):
        city["center"] = {"lat": float(hit["center"]["lat"]), "lon": float(hit["center"]["lon"])}
        changed = True

    # bbox
    cur_bbox = city.get("bbox")
    if overwrite or (cur_bbox is None) or _invalid_bbox(cur_bbox):
        city["bbox"] = _normalize_bbox(hit["bbox"])
        changed = True

    return changed


def main():
    ap = argparse.ArgumentParser(description="Fetch and fill bbox/center for cities config via OSM Nominatim")
    ap.add_argument("in_config", help="Path to input cities config (JSON)")
    ap.add_argument("out_config", help="Path to output cities config (JSON)")
    ap.add_argument("--overwrite", action="store_true", help="Force overwrite existing bbox/center")
    args = ap.parse_args()

    cfg = load_json(args.in_config)
    cache = load_json(CACHE_PATH) if CACHE_PATH.exists() else {}

    updated = 0
    for c in cfg.get("cities", []):
        # Skip if not overwriting and both values valid
        if (not args.overwrite) and (not _invalid_center(c.get("center"))) and (not _invalid_bbox(c.get("bbox"))):
            continue

        key = (c.get("name") or c.get("slug") or "").lower()
        hit = cache.get(key)

        if not hit:
            # Try queries until one resolves
            for q in best_queries(c):
                try:
                    res = fetch_bbox(q)
                    if res:
                        hit = res
                        cache[key] = res
                        break
                except requests.HTTPError as e:
                    if e.response is not None and e.response.status_code == 429:
                        time.sleep(2.0)
                        continue
                except Exception:
                    pass
                finally:
                    time.sleep(1.0)  # be polite to Nominatim

        if hit:
            if upsert_geo(c, hit, overwrite=args.overwrite):
                updated += 1

    if updated:
        save_json(CACHE_PATH, cache)
    save_json(args.out_config, cfg)
    print(f"Saved → {args.out_config} (filled/updated {updated} cities)")


if __name__ == "__main__":
    main()
