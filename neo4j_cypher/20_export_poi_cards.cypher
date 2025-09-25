CALL apoc.export.json.query(
'
MATCH (cp:CanonicalPlace)-[:IN_CITY]->(c:City)

// WV snippets (max 2, non-empty)
OPTIONAL MATCH (cp)-[:HAS_REVIEW]->(w:Review {source:"wikivoyage"})
WITH cp, c, [x IN collect(w) WHERE x.text IS NOT NULL AND trim(x.text) <> "" ][..2] AS wv_top2

// Yelp/Reddit quotes (max 4, non-empty) — PURE CYPHER null removal
OPTIONAL MATCH (cp)-[:HAS_REVIEW]->(yr:Review)
      WHERE yr.source IN ["yelp","reddit"]
WITH cp, c, wv_top2,
     [q IN collect(
        CASE
          WHEN yr.text IS NOT NULL AND trim(yr.text) <> ""
          THEN {src: yr.source, txt: left(yr.text, 220)}
          ELSE NULL
        END
      ) WHERE q IS NOT NULL][..4] AS quotes

RETURN {
  id: cp.id, name: cp.name, city: c.slug, lat: cp.lat, lon: cp.lon,
  wv: [x IN wv_top2 | left(x.text, 600)],
  listings_nearby: cp.listings_nearby,
  popularity_z: coalesce(cp.popularity_blended_z, cp.popularity_listings_z),
  sustainability_popularity_flag: cp.sustainability_popularity_flag,
  quotes: quotes
} AS card
',
'/opt/homebrew/Cellar/neo4j/2025.08.0/libexec/import/poi_cards.json',
{}
);
