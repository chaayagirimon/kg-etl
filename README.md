# Sustainable Tourism KG — Full ETL (single SQLite reviews.db)

This ETL matches your data reality:
- **Yelp + Reddit** from one SQLite DB: `reviews.db`
- **Wikivoyage** from MediaWiki XML dump (2025-08-20)
- **Airbnb** from per-city CSVs (columns you provided)

Pipeline:
1) Build unified `places_raw.csv` from reviews.db + WV XML
2) Stage & assign `city_slug` partitions (Parquet)
3) ER to canonical clusters + links
4) Neo4j exports (+ Cypher)
5) Separate Airbnb branch with Listings/Hosts/Reviews and optional `NEAR` to POIs

## Install
```bash
pip install -r requirements.txt
```

## Inputs

- `./data/raw/reviews.db` (tables: `yelp_businesses`, `yelp_business_reviews`, `reddit_pois` **with latitude/longitude**, `reddit_poi_reviews`)
- Wikivoyage XML (e.g. `./data/raw/enwikivoyage-20250820-pages-articles.xml`)
- Airbnb CSVs under `./data/airbnb/raw/` → `*_listings.csv`, `*_.csv`

## Config

Use your `cities_config.generated.json` (must include `center` & `radius_km` per city).

## Run — POIs
```bash
# 1) Build places_raw.csv from reviews.db + WV XML (any subset OK)
python -m scripts.run_all \
  --config ./cities_config.generated.json \
  --sources-dir ./data/raw \
  --out-dir ./data \
  --build-places \
  --reviews-db ./data/raw/reviews.db \
  --wv-xml ./data/raw/enwikivoyage-20250820-pages-articles.xml \
  --verbose


# 2) Stage
python -m scripts.run_all --config ./cities_config.generated.json --out-dir ./data --stage-only

# 3) ER + exports
python -m scripts.run_all --config ./cities_config.generated.json --out-dir ./data --er-only
```

## Run — Airbnb (separate)
```bash
python -m scripts.run_airbnb   --config ./cities_config.generated.json   --airbnb-dir ./data/airbnb/raw   --out-dir ./data   --places-csv ./data/exports/neo4j/places.csv   --near-threshold-m 300
```

## Neo4j
more to be added on this (TBD)