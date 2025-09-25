# make_clean_cities_bboxes_csv.py
import json, csv
cfg = json.load(open("cities_config.generated.json", "r", encoding="utf-8"))

with open("cities.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow([
        "slug","bbox_min_lat","bbox_min_lon","bbox_max_lat","bbox_max_lon",
        "center_lat","center_lon","radius_km", "name", "country"
    ])
    for c in cfg["cities"]:
        bbox   = c.get("bbox") or [None, None, None, None]  # [min_lat, min_lon, max_lat, max_lon]
        center = c.get("center") or {}
        w.writerow([
            c["slug"], bbox[0], bbox[1], bbox[2], bbox[3],
            center.get("lat"), center.get("lon"), c.get("radius_km"), 
            c.get("name"), c.get("country")
        ])
