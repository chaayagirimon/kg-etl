// Hosts
LOAD CSV WITH HEADERS FROM 'file:///airbnb/hosts.csv' AS row
CALL (row) {
  WITH row
  WHERE row.host_id IS NOT NULL AND row.host_id <> ''
  MERGE (h:Host {id: row.host_id})
  SET  h.name                 = row.host_name,
       h.is_superhost         = CASE row.host_is_superhost WHEN 't' THEN true WHEN 'True' THEN true WHEN '1' THEN true WHEN 'TRUE' THEN true ELSE false END,
       h.listings_count       = CASE WHEN row.host_listings_count <> '' THEN toInteger(row.host_listings_count) ELSE NULL END,
       h.total_listings_count = CASE WHEN row.host_total_listings_count <> '' THEN toInteger(row.host_total_listings_count) ELSE NULL END
} IN TRANSACTIONS OF 5000 ROWS;

// HOSTS edges (from listings)
LOAD CSV WITH HEADERS FROM 'file:///airbnb/listings.csv' AS row
CALL (row) {
  WITH row
  WHERE row.listing_id IS NOT NULL AND row.host_id IS NOT NULL
  MATCH (l:Listing {id: row.listing_id})
  MERGE (h:Host {id: row.host_id})
  MERGE (h)-[:HOSTS]->(l)
} IN TRANSACTIONS OF 5000 ROWS;
