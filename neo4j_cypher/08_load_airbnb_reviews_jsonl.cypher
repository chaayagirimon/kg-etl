// 08_load_airbnb_reviews_jsonl.cypher
// Neo4j 5 + APOC Core required.
// Loads Airbnb reviews, links Listing -> Review, materializes Listing.cp_id via
// NEAR(SourcePlace)->VARIANT_OF(CanonicalPlace), then lifts reviews to CanonicalPlace.
// Uses small batches to avoid transaction memory errors.

// -----------------------------------------------------------------------------
// Indexes (idempotent)
// -----------------------------------------------------------------------------
CREATE INDEX review_id IF NOT EXISTS FOR (r:Review) ON (r.id);
CREATE INDEX review_source IF NOT EXISTS FOR (r:Review) ON (r.source);
CREATE INDEX listing_id IF NOT EXISTS FOR (l:Listing) ON (l.id);
CREATE INDEX sourceplace_id IF NOT EXISTS FOR (sp:SourcePlace) ON (sp.id);
CREATE INDEX canonicalplace_id IF NOT EXISTS FOR (cp:CanonicalPlace) ON (cp.id);

// -----------------------------------------------------------------------------
// 1) Load Airbnb reviews JSONL and link Listing -> Review
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    CALL apoc.load.json('file:///airbnb/reviews.jsonl') YIELD value
    WITH value
    WHERE value.review_id IS NOT NULL AND value.review_id <> ''
      AND value.listing_id IS NOT NULL AND value.listing_id <> ''
    RETURN value
  ",
  "
    MERGE (rv:Review {id: value.review_id})
    SET  rv.source     = 'airbnb',
         rv.rating     = CASE WHEN value.rating IS NULL THEN rv.rating ELSE toInteger(value.rating) END,
         rv.text       = coalesce(value.text, rv.text),
         rv.scraped_at = coalesce(value.scraped_at, rv.scraped_at),
         rv.listing_id = coalesce(value.listing_id, rv.listing_id)

    WITH rv, value
    MATCH (l:Listing {id: value.listing_id})
    MERGE (l)-[:HAS_REVIEW]->(rv)
  ",
  {batchSize: 20000, parallel: false}
);

// -----------------------------------------------------------------------------
// 2) Materialize canonical place id on Listing via NEAR->VARIANT_OF (nearest by meters)
//    Adjust $maxMeters as needed; pass NULL to ignore distance.
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)-[n:NEAR]->(sp:SourcePlace)-[:VARIANT_OF]->(cp:CanonicalPlace)
    WHERE $maxMeters IS NULL OR n.meters IS NULL OR n.meters <= $maxMeters
    WITH l, cp.id AS cp_id, coalesce(n.meters, 0.0) AS meters
    ORDER BY meters ASC
    WITH l, collect(cp_id)[0] AS cp_id      // pick nearest cp_id per listing
    RETURN l, cp_id
  ",
  "
    SET l.cp_id = cp_id
  ",
  {batchSize: 5000, parallel: false, params: {maxMeters: 300.0}}
);

// -----------------------------------------------------------------------------
// 3) Lift Airbnb reviews to CanonicalPlace using materialized cp_id
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)-[:HAS_REVIEW]->(r:Review {source:'airbnb'})
    WHERE l.cp_id IS NOT NULL
    RETURN r, l.cp_id AS cp_id
  ",
  "
    MATCH (cp:CanonicalPlace {id: cp_id})
    MERGE (cp)-[:HAS_REVIEW]->(r)
  ",
  {batchSize: 20000, parallel: false}
);

// -----------------------------------------------------------------------------
// 4) Fallback: if a listing has no cp_id, attach reviews to nearest CP in same city
//    within $fallbackMaxMeters. Comment out this block if you want NEAR-only lifting.
//    (Neo4j 5: use point.distance; use IS NULL / IS NOT NULL)
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    MATCH (l:Listing)-[:HAS_REVIEW]->(r:Review {source:'airbnb'})
    WHERE l.cp_id IS NULL
      AND l.lat IS NOT NULL AND l.lon IS NOT NULL
    MATCH (l)-[:IN_CITY]->(c:City)
    RETURN l, r, c
  ",
  "
    MATCH (cp:CanonicalPlace)-[:IN_CITY]->(c)
    WHERE cp.lat IS NOT NULL AND cp.lon IS NOT NULL
    WITH l, r, cp,
         point({latitude:l.lat,  longitude:l.lon})  AS pl,
         point({latitude:cp.lat, longitude:cp.lon}) AS pcp
    WITH r, cp, point.distance(pl, pcp) AS meters   // <-- Neo4j 5
    WHERE meters <= $fallbackMaxMeters
    ORDER BY meters ASC
    WITH r, collect({cp:cp, m:meters}) AS cands
    WHERE size(cands) > 0
    WITH r, cands[0].cp AS best
    MERGE (best)-[:HAS_REVIEW]->(r)
  ",
  {batchSize: 1000, parallel: false, params: {fallbackMaxMeters: 200.0}}
);

// -----------------------------------------------------------------------------
// 5) Clean up helper property (optional)
// -----------------------------------------------------------------------------
MATCH (l:Listing) WHERE l.cp_id IS NOT NULL REMOVE l.cp_id;

// -----------------------------------------------------------------------------
// 6) Quick counts
// -----------------------------------------------------------------------------
MATCH (:Listing)-[:HAS_REVIEW]->(:Review {source:'airbnb'})
RETURN count(*) AS airbnb_reviews_linked_to_listings;

MATCH (cp:CanonicalPlace)-[:HAS_REVIEW]->(:Review {source:'airbnb'})
RETURN count(*) AS airbnb_reviews_linked_to_canonical;
