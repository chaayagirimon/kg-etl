// 05_load_place_reviews_jsonl.cypher
// Requires APOC Core (apoc.load.json, apoc.periodic.iterate).
// Neo4j 5 compatible. Idempotent.

// -----------------------------------------------------------------------------
// Indexes (safe to re-run)
// -----------------------------------------------------------------------------
CREATE INDEX review_id IF NOT EXISTS FOR (r:Review) ON (r.id);
CREATE INDEX sourceplace_id IF NOT EXISTS FOR (sp:SourcePlace) ON (sp.id);
CREATE INDEX sourceplace_place_id IF NOT EXISTS FOR (sp:SourcePlace) ON (sp.place_id);

// -----------------------------------------------------------------------------
// Load JSONL and attach to SourcePlace
// - We also persist rv.place_id and rv.city_slug for diagnostics and fallback joins.
// - We link to SourcePlace by either property name: {id: ...} or {place_id: ...}.
// -----------------------------------------------------------------------------
CALL apoc.periodic.iterate(
  "
    CALL apoc.load.json('file:///place_reviews.jsonl') YIELD value
    // filter out lines without a review_id (MERGE cannot use null)
    WITH value
    WHERE value.review_id IS NOT NULL AND value.review_id <> ''
    RETURN value
  ",
  "
    // Upsert Review
    MERGE (rv:Review {id: value.review_id})
    SET  rv.source     = coalesce(value.source,     rv.source),
         rv.rating     = CASE WHEN value.rating IS NULL THEN rv.rating ELSE toInteger(value.rating) END,
         rv.text       = coalesce(value.text,       rv.text),
         rv.scraped_at = coalesce(value.scraped_at, rv.scraped_at),
         rv.place_id   = coalesce(value.place_id,   rv.place_id),  // keep for debugging/joins
         rv.city_slug  = coalesce(value.city_slug,  rv.city_slug)

    // Try to find SourcePlace (first by id, then by place_id)
    WITH rv, value
    OPTIONAL MATCH (sp1:SourcePlace {id: value.place_id})
    WITH rv, coalesce(sp1, NULL) AS sp, value
    OPTIONAL MATCH (sp2:SourcePlace {place_id: value.place_id})
    WITH rv, coalesce(sp, sp2) AS sp

    // Create the relationship if SourcePlace was found
    FOREACH (_ IN CASE WHEN sp IS NULL THEN [] ELSE [1] END |
      MERGE (sp)-[:HAS_REVIEW]->(rv)
    )
  ",
  {batchSize: 20000, parallel: false, concurrency: 1}
);

// -----------------------------------------------------------------------------
// Optional: quick counts (purely informational; comment out if you prefer)
// -----------------------------------------------------------------------------
MATCH (r:Review) RETURN count(r) AS reviews_total;

MATCH (:SourcePlace)-[:HAS_REVIEW]->(:Review) RETURN count(*) AS sp_review_links;

MATCH (:CanonicalPlace)-[:HAS_REVIEW]->(:Review) RETURN count(*) AS cp_review_links;  // will grow after 11_*.cypher
