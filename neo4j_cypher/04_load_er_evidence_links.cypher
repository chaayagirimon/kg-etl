LOAD CSV WITH HEADERS FROM 'file:///place_links.csv' AS row
CALL (row) {
  WITH row
  WHERE row.a IS NOT NULL AND row.b IS NOT NULL
  MATCH (a:SourcePlace {id: row.a})
  MATCH (b:SourcePlace {id: row.b})
  MERGE (a)-[r:SAME_AS_LINK]->(b)
    ON CREATE SET
      r.name_sim = CASE WHEN row.name_sim <> '' THEN toFloat(row.name_sim) ELSE NULL END,
      r.meters   = CASE WHEN row.meters   <> '' THEN toFloat(row.meters)   ELSE NULL END,
      r.city_slug= row.city_slug
} IN TRANSACTIONS OF 5000 ROWS;
