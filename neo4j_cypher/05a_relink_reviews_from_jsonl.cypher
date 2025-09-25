// 05a_relink_reviews_from_jsonl.cypher
// Neo4j 5+; needs APOC Core for apoc.load.json + apoc.periodic.iterate

// Safety: indexes/constraints
CREATE CONSTRAINT review_id_unique IF NOT EXISTS
FOR (r:Review) REQUIRE (r.id) IS UNIQUE;

CREATE INDEX sourceplace_id IF NOT EXISTS
FOR (p:SourcePlace) ON (p.id);

CREATE INDEX canonicalplace_id IF NOT EXISTS
FOR (cp:CanonicalPlace) ON (cp.id);

// Iterate JSONL and fix links for existing reviews
CALL apoc.periodic.iterate(
  "CALL apoc.load.json('file:///place_reviews.jsonl') YIELD value RETURN value",
  "
   // Match the already-created Review by id
   MATCH (rv:Review {id: value.review_id})

   // Add useful props for debugging/audit
   SET  rv.source     = coalesce(value.source, rv.source),
        rv.rating     = CASE WHEN value.rating IS NULL THEN rv.rating ELSE toInteger(value.rating) END,
        rv.text       = coalesce(value.text, rv.text),
        rv.scraped_at = coalesce(value.scraped_at, rv.scraped_at),
        rv.place_id   = coalesce(value.place_id, rv.place_id),
        rv.city_slug  = coalesce(value.city_slug, rv.city_slug)

   // Link to SourcePlace if present
   WITH rv, value
   OPTIONAL MATCH (sp:SourcePlace {id: value.place_id})
   FOREACH (_ IN CASE WHEN sp IS NULL THEN [] ELSE [1] END |
     MERGE (sp)-[:HAS_REVIEW]->(rv)
   )

   // Also link to CanonicalPlace if sp→cp mapping exists (any of these rels)
   WITH rv, sp
   OPTIONAL MATCH (sp)-[:SAME_AS|:CANONICAL|:IS_CANONICAL|:IN_CANONICAL|:ALIGNED_WITH]->(cp:CanonicalPlace)
   FOREACH (_ IN CASE WHEN cp IS NULL THEN [] ELSE [1] END |
     MERGE (rv)-[:ABOUT]->(cp)
   )
  ",
  {batchSize:20000, parallel:false}
);

// Report a quick summary
CALL {
  MATCH (r:Review)-[:HAS_REVIEW]->(:SourcePlace)
  RETURN count(*) AS linked_to_source
}
CALL {
  MATCH (r:Review)-[:ABOUT]->(:CanonicalPlace)
  RETURN count(*) AS linked_to_canonical
}
RETURN linked_to_source, linked_to_canonical;
