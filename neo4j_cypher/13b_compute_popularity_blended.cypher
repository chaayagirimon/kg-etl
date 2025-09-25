// --- 1) Per-CP review count (yelp+reddit) ---
MATCH (cp:CanonicalPlace)
OPTIONAL MATCH (cp)-[:HAS_REVIEW]->(r:Review)
  WHERE r.source IN ["yelp","reddit"] AND r.text IS NOT NULL AND trim(r.text) <> ""
WITH cp, count(r) AS revs
SET cp.reviews_count = revs;

// --- 2) Compute per-city z-scores for listings_nearby and reviews_count ---
MATCH (c:City)<-[:IN_CITY]-(cp:CanonicalPlace)
WITH c,
     collect({cp:cp,
              ln: coalesce(cp.listings_nearby, 0),
              rc: coalesce(cp.reviews_count, 0)}) AS rows

// helper stats for ln
WITH c, rows, size(rows) AS n,
     reduce(sln=0.0, r IN rows | sln + r.ln) AS sum_ln,
     reduce(ssln=0.0, r IN rows | ssln + r.ln*r.ln) AS sumsq_ln,
     reduce(src=0.0, r IN rows | src + r.rc) AS sum_rc,
     reduce(ssrc=0.0, r IN rows | ssrc + r.rc*r.rc) AS sumsq_rc

WITH c, rows, n,
     (sum_ln*1.0)/n AS mu_ln,
     CASE WHEN n <= 1 THEN 0.0 ELSE sqrt((sumsq_ln - (sum_ln*sum_ln)/n)/(n-1)) END AS sigma_ln,
     (sum_rc*1.0)/n AS mu_rc,
     CASE WHEN n <= 1 THEN 0.0 ELSE sqrt((sumsq_rc - (sum_rc*sum_rc)/n)/(n-1)) END AS sigma_rc
UNWIND rows AS row
WITH row.cp AS cp, row.ln AS ln, row.rc AS rc, mu_ln, sigma_ln, mu_rc, sigma_rc

// handle zero-sigma safely
WITH cp,
     CASE WHEN sigma_ln = 0.0 OR sigma_ln IS NULL THEN 0.0 ELSE (ln - mu_ln)/sigma_ln END AS z_ln,
     CASE WHEN sigma_rc = 0.0 OR sigma_rc IS NULL THEN 0.0 ELSE (rc - mu_rc)/sigma_rc END AS z_rc

// --- 3) Blend (tune weights as needed) ---
WITH cp, z_ln, z_rc, (0.6 * z_ln + 0.4 * z_rc) AS z_blend
SET cp.popularity_listings_z  = z_ln,
    cp.popularity_reviews_z   = z_rc,
    cp.popularity_blended_z   = z_blend;

// --- 4) Flag from blended z if available (fallback to listings z) ---
MATCH (cp:CanonicalPlace)
WITH cp, coalesce(cp.popularity_blended_z, cp.popularity_listings_z, 0.0) AS z
SET cp.sustainability_popularity_flag =
  CASE WHEN z >=  1.0 THEN "high"
       WHEN z >=  0.0 THEN "medium"
       ELSE              "low" END;
