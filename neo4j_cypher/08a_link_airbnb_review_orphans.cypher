// 08a_link_airbnb_review_orphans.cypher
// Idempotent helper that only relinks Airbnb reviews that are still missing a
// CanonicalPlace relationship. Useful when the main 08 load has already run and
// you merely need to catch up orphans without reprocessing everything.
// Neo4j 5 + APOC Core required.

// -----------------------------------------------------------------------------
// Index safety (only run if needed; they are idempotent)
// -----------------------------------------------------------------------------
CREATE INDEX listing_id IF NOT EXISTS FOR (l:Listing) ON (l.id);
CREATE INDEX canonicalplace_id IF NOT EXISTS FOR (cp:CanonicalPlace) ON (cp.id);
CREATE INDEX review_source IF NOT EXISTS FOR (r:Review) ON (r.source);

// -----------------------------------------------------------------------------
// 1) Materialize cp_id via NEAR->VARIANT_OF only for listings still missing it
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)
    WHERE l.cp_id IS NULL
    MATCH (l)-[n:NEAR]->(sp:SourcePlace)-[:VARIANT_OF]->(cp:CanonicalPlace)
    WHERE $maxMeters IS NULL OR n.meters IS NULL OR n.meters <= $maxMeters
    WITH l, cp.id AS cp_id, coalesce(n.meters, 0.0) AS meters
    ORDER BY meters ASC
    WITH l, collect(cp_id)[0] AS cp_id
    RETURN l, cp_id
  ",
  "
    SET l.cp_id = cp_id
  ",
  {batchSize: 2000, parallel: false, params: {maxMeters: 2000.0}}
);

// -----------------------------------------------------------------------------
// 2) Fallback: use geometry within city for any remaining listings
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)
    WHERE l.cp_id IS NULL
      AND l.lat IS NOT NULL AND l.lon IS NOT NULL
    MATCH (l)-[:IN_CITY]->(c:City)
    RETURN l, c
  ",
  "
    WITH l, c, point({latitude:l.lat, longitude:l.lon}) AS pl
    CALL {
      WITH l, c, pl
      MATCH (cp:CanonicalPlace)-[:IN_CITY]->(c)
      WHERE cp.lat IS NOT NULL AND cp.lon IS NOT NULL
      WITH cp, point.distance(pl, point({latitude:cp.lat, longitude:cp.lon})) AS meters
      WHERE meters <= $fallbackMaxMeters
      ORDER BY meters ASC
      LIMIT 1
      RETURN cp
    }
    WITH l, cp
    WHERE cp IS NOT NULL
    SET l.cp_id = cp.id
  ",
  {batchSize: 100, parallel: false, params: {fallbackMaxMeters: 2000.0}}
);

// -----------------------------------------------------------------------------
// 3) Link only the still-orphaned reviews (no CanonicalPlace HAS_REVIEW yet)
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)-[:HAS_REVIEW]->(r:Review {source:'airbnb'})
    WHERE l.cp_id IS NOT NULL
      AND NOT (r)<-[:HAS_REVIEW]-(:CanonicalPlace)
    RETURN r, l.cp_id AS cp_id
  ",
  "
    MATCH (cp:CanonicalPlace {id: cp_id})
    MERGE (cp)-[:HAS_REVIEW]->(r)
  ",
  {batchSize: 5000, parallel: false}
);

