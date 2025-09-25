// -- Canonical coordinates from SourcePlace (priority: yelp > wikivoyage > reddit > any) --
MATCH (cp:CanonicalPlace)
REMOVE cp.lat, cp.lon;

MATCH (cp:CanonicalPlace)<-[:VARIANT_OF]-(sp:SourcePlace {source:'yelp'})
WITH cp, avg(sp.lat) AS lat, avg(sp.lon) AS lon
SET cp.lat = coalesce(cp.lat, lat),
    cp.lon = coalesce(cp.lon, lon);

MATCH (cp:CanonicalPlace)<-[:VARIANT_OF]-(sp:SourcePlace {source:'wikivoyage'})
WITH cp, avg(sp.lat) AS lat, avg(sp.lon) AS lon
SET cp.lat = coalesce(cp.lat, lat),
    cp.lon = coalesce(cp.lon, lon);

MATCH (cp:CanonicalPlace)<-[:VARIANT_OF]-(sp:SourcePlace {source:'reddit'})
WITH cp, avg(sp.lat) AS lat, avg(sp.lon) AS lon
SET cp.lat = coalesce(cp.lat, lat),
    cp.lon = coalesce(cp.lon, lon);

MATCH (cp:CanonicalPlace)<-[:VARIANT_OF]-(sp:SourcePlace)
WITH cp, avg(sp.lat) AS lat, avg(sp.lon) AS lon
SET cp.lat = coalesce(cp.lat, lat),
    cp.lon = coalesce(cp.lon, lon);

// -- Lift Yelp/Reddit/WV reviews to CanonicalPlace (idempotent) --
MATCH (sp:SourcePlace)-[:VARIANT_OF]->(cp:CanonicalPlace)
MATCH (sp)-[:HAS_REVIEW]->(r:Review)
WHERE r.source IN ['yelp','reddit','wikivoyage']
MERGE (cp)-[:HAS_REVIEW]->(r);
