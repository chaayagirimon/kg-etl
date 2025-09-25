from pathlib import Path
import pandas as pd
from typing import Optional
from .yelp_sqlite import load_yelp_sqlite
from .reddit_sqlite import load_reddit_sqlite, enrich_reddit_with_coords
from .wikivoyage_xml import load_wikivoyage_xml

def build_places_raw_sqlite(sources_dir: str, out_csv: str, reviews_db: Optional[str], wv_xml: Optional[str]) -> pd.DataFrame:
    frames = []
    yelp = load_yelp_sqlite(reviews_db) if reviews_db else None
    reddit = load_reddit_sqlite(reviews_db) if reviews_db else None
    wv = load_wikivoyage_xml(wv_xml) if wv_xml else None

    if yelp is not None and not yelp.empty:
        frames.append(yelp)
    if wv is not None and not wv.empty:
        frames.append(wv)

    if reddit is not None and not reddit.empty:
        reddit_enriched = enrich_reddit_with_coords(reddit, yelp, wv, sim_threshold=0.86)
        frames.append(reddit_enriched)

    if not frames:
        raise SystemExit("No sources found. Provide reviews.db and/or WV XML.")

    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["lat","lon","name"]).reset_index(drop=True)

    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[unify] wrote {len(out)} rows -> {out_csv}")
    return out
