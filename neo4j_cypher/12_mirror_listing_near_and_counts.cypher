// Clear old stats/flags
MATCH (cp:CanonicalPlace)
REMOVE cp.popularity_listings_mu,
       cp.popularity_listings_sigma,
       cp.popularity_listings_z,
       cp.sustainability_popularity_flag;

// Per-city μ/σ over listings_nearby, then z-score and flag
MATCH (c:City)<-[:IN_CITY]-(cp:CanonicalPlace)
WITH c, collect({cp:cp, ln:coalesce(cp.listings_nearby,0)}) AS rows
WITH c, rows, size(rows) AS n,
     reduce(sum_ln=0.0,  r IN rows | sum_ln + r.ln) AS sum_ln,
     reduce(sumsq=0.0,   r IN rows | sumsq + r.ln*r.ln) AS sumsq
WITH c, rows, n,
     (sum_ln*1.0)/n AS mu,
     CASE WHEN n <= 1 THEN 0.0
          ELSE sqrt( (sumsq - (sum_ln*sum_ln)/n) / (n-1) ) END AS sigma
UNWIND rows AS row
WITH row.cp AS cp, row.ln AS ln, mu, sigma
SET cp.popularity_listings_mu    = mu,
    cp.popularity_listings_sigma = sigma,
    cp.popularity_listings_z     = CASE WHEN sigma = 0.0 OR sigma IS NULL
                                        THEN 0.0 ELSE (ln - mu)/sigma END;

MATCH (cp:CanonicalPlace)
WITH cp, coalesce(cp.popularity_listings_z, 0.0) AS z
SET cp.sustainability_popularity_flag =
  CASE WHEN z >=  1.0 THEN 'high'
       WHEN z >=  0.0 THEN 'medium'
       ELSE               'low' END;
