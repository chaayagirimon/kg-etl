// Constraints & indexes (run once)
CREATE CONSTRAINT city_slug IF NOT EXISTS
FOR (c:City) REQUIRE c.slug IS UNIQUE;

CREATE CONSTRAINT source_place_id IF NOT EXISTS
FOR (p:SourcePlace) REQUIRE p.id IS UNIQUE;

CREATE CONSTRAINT canonical_place_id IF NOT EXISTS
FOR (c:CanonicalPlace) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT listing_id IF NOT EXISTS
FOR (l:Listing) REQUIRE l.id IS UNIQUE;

CREATE CONSTRAINT host_id IF NOT EXISTS
FOR (h:Host) REQUIRE h.id IS UNIQUE;

CREATE CONSTRAINT review_id IF NOT EXISTS
FOR (r:Review) REQUIRE r.id IS UNIQUE;

CREATE INDEX source_place_city IF NOT EXISTS
FOR (p:SourcePlace) ON (p.city_slug);

CREATE INDEX canonical_place_city IF NOT EXISTS
FOR (c:CanonicalPlace) ON (c.city_slug);
