LOAD CSV WITH HEADERS FROM 'file:///airbnb/listing_near_place.csv' AS row
CALL (row) {
  WITH row
  MATCH (l:Listing {id: row.listing_id})
  MATCH (p:SourcePlace {id: row.place_id})
  MERGE (l)-[r:NEAR]->(p)
    ON CREATE SET r.meters   = CASE WHEN row.meters <> '' THEN toFloat(row.meters) ELSE NULL END,
                  r.city_slug= row.city_slug
} IN TRANSACTIONS OF 5000 ROWS;
