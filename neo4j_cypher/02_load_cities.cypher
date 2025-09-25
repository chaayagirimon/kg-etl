// 02_load_cities.cypher  (Neo4j 5+; robust to empty fields)

// 0) Ensure unique slug
CREATE CONSTRAINT city_slug_unique IF NOT EXISTS
FOR (c:City) REQUIRE (c.slug) IS UNIQUE;

// 1) Upsert City nodes from cities.csv in batches
CALL {
  LOAD CSV WITH HEADERS FROM 'file:///cities.csv' AS row

  WITH
    // Clean + cast
    trim(row.slug) AS slug,
    CASE WHEN row.bbox_min_lat  IS NULL OR row.bbox_min_lat  = '' THEN NULL ELSE toFloat(row.bbox_min_lat)  END AS bbox_min_lat,
    CASE WHEN row.bbox_min_lon  IS NULL OR row.bbox_min_lon  = '' THEN NULL ELSE toFloat(row.bbox_min_lon)  END AS bbox_min_lon,
    CASE WHEN row.bbox_max_lat  IS NULL OR row.bbox_max_lat  = '' THEN NULL ELSE toFloat(row.bbox_max_lat)  END AS bbox_max_lat,
    CASE WHEN row.bbox_max_lon  IS NULL OR row.bbox_max_lon  = '' THEN NULL ELSE toFloat(row.bbox_max_lon)  END AS bbox_max_lon,
    CASE WHEN row.center_lat    IS NULL OR row.center_lat    = '' THEN NULL ELSE toFloat(row.center_lat)    END AS center_lat,
    CASE WHEN row.center_lon    IS NULL OR row.center_lon    = '' THEN NULL ELSE toFloat(row.center_lon)    END AS center_lon,
    CASE WHEN row.radius_km     IS NULL OR row.radius_km     = '' THEN NULL ELSE toFloat(row.radius_km)     END AS radius_km,
    CASE WHEN row.name          IS NULL OR row.name          = '' THEN NULL ELSE row.name                    END AS name,
    CASE WHEN row.country       IS NULL OR row.country       = '' THEN NULL ELSE row.country                 END AS country

  // IMPORTANT: never MERGE with a NULL/empty slug
  WHERE slug IS NOT NULL AND slug <> ''

  MERGE (c:City {slug: slug})

  // Only set a property when we have a non-NULL value (prevents NULL clobbering)
  FOREACH (_ IN CASE WHEN name          IS NULL THEN [] ELSE [1] END | SET c.name         = name)
  FOREACH (_ IN CASE WHEN country       IS NULL THEN [] ELSE [1] END | SET c.country      = country)
  FOREACH (_ IN CASE WHEN bbox_min_lat  IS NULL THEN [] ELSE [1] END | SET c.bbox_min_lat = bbox_min_lat)
  FOREACH (_ IN CASE WHEN bbox_min_lon  IS NULL THEN [] ELSE [1] END | SET c.bbox_min_lon = bbox_min_lon)
  FOREACH (_ IN CASE WHEN bbox_max_lat  IS NULL THEN [] ELSE [1] END | SET c.bbox_max_lat = bbox_max_lat)
  FOREACH (_ IN CASE WHEN bbox_max_lon  IS NULL THEN [] ELSE [1] END | SET c.bbox_max_lon = bbox_max_lon)
  FOREACH (_ IN CASE WHEN center_lat    IS NULL THEN [] ELSE [1] END | SET c.center_lat   = center_lat)
  FOREACH (_ IN CASE WHEN center_lon    IS NULL THEN [] ELSE [1] END | SET c.center_lon   = center_lon)
  FOREACH (_ IN CASE WHEN radius_km     IS NULL THEN [] ELSE [1] END | SET c.radius_km    = radius_km)
} IN TRANSACTIONS OF 1000 ROWS;

// 2) Normalize bbox bounds if min/max were swapped
MATCH (c:City)
WITH c,
     toFloat(c.bbox_min_lat) AS minla, toFloat(c.bbox_max_lat) AS maxla,
     toFloat(c.bbox_min_lon) AS minlo, toFloat(c.bbox_max_lon) AS maxlo
SET c.bbox_min_lat = CASE WHEN minla IS NULL OR maxla IS NULL THEN c.bbox_min_lat
                          WHEN minla <= maxla THEN minla ELSE maxla END,
    c.bbox_max_lat = CASE WHEN minla IS NULL OR maxla IS NULL THEN c.bbox_max_lat
                          WHEN minla <= maxla THEN maxla ELSE minla END,
    c.bbox_min_lon = CASE WHEN minlo IS NULL OR maxlo IS NULL THEN c.bbox_min_lon
                          WHEN minlo <= maxlo THEN minlo ELSE maxlo END,
    c.bbox_max_lon = CASE WHEN minlo IS NULL OR maxlo IS NULL THEN c.bbox_max_lon
                          WHEN minlo <= maxlo THEN maxlo ELSE minlo END;
