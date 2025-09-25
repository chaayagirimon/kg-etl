// Attach Yelp/Reddit/WV reviews to CanonicalPlace (idempotent)
MATCH (sp:SourcePlace)-[:VARIANT_OF]->(cp:CanonicalPlace)
MATCH (sp)-[:HAS_REVIEW]->(r:Review)
WHERE r.source IN ['yelp','reddit','wikivoyage']
MERGE (cp)-[:HAS_REVIEW]->(r);
