from pathlib import Path
import argparse
from etl.config_loader import load_config
from etl.reviews import extract_reviews_sqlite

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to cities_config.generated.json")
    ap.add_argument("--reviews-db", required=True, help="Path to reviews.db (Yelp+Reddit)")
    ap.add_argument("--out-dir", default="./data", help="Output directory root")
    args = ap.parse_args()

    cfg = load_config(args.config)
    neo_df, rag_path = extract_reviews_sqlite(args.reviews_db, cfg, args.out_dir)
    # print(f"Wrote Neo4j reviews CSV to {Path(args.out_dir)/'exports'/'neo4j'/'place_reviews.csv'}")
    print(f"Wrote RAG JSONL to {rag_path}")

if __name__ == "__main__":
    main()
