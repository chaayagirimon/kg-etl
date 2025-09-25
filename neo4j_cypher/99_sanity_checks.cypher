MATCH (c:City) RETURN count(c) AS cities;
MATCH (p:SourcePlace) RETURN count(p) AS source_places;
MATCH (k:CanonicalPlace) RETURN count(k) AS canonical_places;
MATCH ()-[:VARIANT_OF]->() RETURN count(*) AS variant_edges;
MATCH ()-[:SAME_AS_LINK]->() RETURN count(*) AS evidence_links;
MATCH (l:Listing)-[:IN_CITY]->(c:City) RETURN c.slug, count(*) AS listings ORDER BY listings DESC LIMIT 10;
MATCH (l:Listing)-[r:NEAR]->(p:SourcePlace) RETURN p.city_slug, count(r) AS near_edges ORDER BY near_edges DESC LIMIT 10;
MATCH (p:SourcePlace)-[:HAS_REVIEW]->(rv:Review) RETURN p.source, count(rv) AS reviews ORDER BY reviews DESC;
