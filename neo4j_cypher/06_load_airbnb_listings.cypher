LOAD CSV WITH HEADERS FROM 'file:///airbnb/listings.csv' AS row
CALL (row) {
  WITH row
  WHERE row.listing_id IS NOT NULL AND row.listing_id <> ''
  MERGE (l:Listing {id: row.listing_id})
  SET  l.name         = row.name,
       l.lat          = CASE WHEN row.lat <> '' THEN toFloat(row.lat) ELSE NULL END,
       l.lon          = CASE WHEN row.lon <> '' THEN toFloat(row.lon) ELSE NULL END,
       l.room_type    = row.room_type,
       l.accommodates = CASE WHEN row.accommodates <> '' THEN toInteger(row.accommodates) ELSE NULL END,
       l.bedrooms     = CASE WHEN row.bedrooms <> '' THEN toFloat(row.bedrooms) ELSE NULL END,
       l.bathrooms    = CASE WHEN row.bathrooms <> '' THEN toFloat(row.bathrooms) ELSE NULL END,
       l.price        = CASE WHEN row.price <> '' THEN toFloat(row.price) ELSE NULL END,
       l.city_slug    = row.city_slug
} IN TRANSACTIONS OF 5000 ROWS;
