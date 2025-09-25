LOAD CSV WITH HEADERS FROM 'file:///airbnb/listing_city.csv' AS row
CALL (row) {
  WITH row
  MATCH (l:Listing {id: row.listing_id})
  MATCH (c:City {slug: row.city_slug})
  MERGE (l)-[:IN_CITY]->(c)
} IN TRANSACTIONS OF 5000 ROWS;
