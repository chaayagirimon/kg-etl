// ---------- PASS 1: SourcePlace + IN_CITY (from places.csv) ----------
LOAD CSV WITH HEADERS FROM 'file:///places.csv' AS row
WITH row
WHERE row.place_id IS NOT NULL AND row.place_id <> ''
CALL {
  WITH row  // importing WITH: pass variables only
  MERGE (p:SourcePlace {id: row.place_id})
  SET  p.source    = row.source,
       p.name      = row.name,
       p.lat       = CASE WHEN row.lat <> '' THEN toFloat(row.lat) ELSE NULL END,
       p.lon       = CASE WHEN row.lon <> '' THEN toFloat(row.lon) ELSE NULL END,
       p.city_slug = row.city_slug,
       p.address   = CASE WHEN row.address IS NOT NULL AND row.address <> '' THEN row.address ELSE p.address END,
       p.type      = CASE WHEN row.type    IS NOT NULL AND row.type    <> '' THEN row.type    ELSE p.type    END
  WITH row, p
  MATCH (city:City {slug: row.city_slug})
  MERGE (p)-[:IN_CITY]->(city)
} IN TRANSACTIONS OF 2000 ROWS;

// ---------- PASS 2: CanonicalPlace (city-scoped) + VARIANT_OF (from place_canonical_map.csv) ----------
LOAD CSV WITH HEADERS FROM 'file:///place_canonical_map.csv' AS row
WITH row
WHERE row.canonical_id IS NOT NULL AND row.canonical_id <> ''
  AND row.source_place_id IS NOT NULL AND row.source_place_id <> ''
  AND row.city_slug IS NOT NULL AND row.city_slug <> ''
CALL {
  WITH row  // importing WITH: pass variables only
  MERGE (cp:CanonicalPlace {id: row.canonical_id})
    ON CREATE SET cp.name = row.canonical_name
  WITH row, cp
  MATCH (city:City {slug: row.city_slug})
  MERGE (cp)-[:IN_CITY]->(city)
  WITH row, cp
  MATCH (sp:SourcePlace {id: row.source_place_id})
  MERGE (sp)-[:VARIANT_OF]->(cp)
} IN TRANSACTIONS OF 2000 ROWS;
