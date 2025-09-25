#!/usr/bin/env bash
# run_all.sh — execute all (or a subset of) Cypher scripts with logging.

set -Eeuo pipefail

# ---- Config (override via environment) ---------------------------------------
NEO4J_URI="${NEO4J_URI:-bolt://localhost:7687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASS="${NEO4J_PASS:-MyStrongP@ssw0rd}"
NEO4J_DB="${NEO4J_DB:-neo4j}"

# Which files to run (glob). Override e.g. GLOB='neo4j_cypher/0[3-6]_*.cypher'
GLOB="${GLOB:-neo4j_cypher/*.cypher}"

# Log directory (timestamped)
LOGDIR="${LOGDIR:-logs/$(date +'%Y-%m-%d')}"
mkdir -p "$LOGDIR"

# ---- Checks ------------------------------------------------------------------
command -v cypher-shell >/dev/null 2>&1 || {
  echo "ERROR: cypher-shell not found in PATH." >&2
  exit 127
}

# ---- Run ---------------------------------------------------------------------
echo "Neo4j: $NEO4J_URI  DB: $NEO4J_DB  User: $NEO4J_USER"
echo "Files: $GLOB"
echo "Logs : $LOGDIR"
echo

shopt -s nullglob
files=( $GLOB )
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "No files match '$GLOB' — nothing to run." >&2
  exit 1
fi

i=0
for f in "${files[@]}"; do
  ((i++))
  base="$(basename "$f")"
  log="$LOGDIR/${i}_$base.log"

  echo ">>> [$i/${#files[@]}] Running $f"
  {
    echo "=== $(date) ==="
    echo "FILE: $f"
    echo "URI : $NEO4J_URI  DB: $NEO4J_DB  USER: $NEO4J_USER"
    echo "------------------------------------------------------------------------"
    cypher-shell -a "$NEO4J_URI" -u "$NEO4J_USER" -p "$NEO4J_PASS" -d "$NEO4J_DB" -f "$f"
    echo "------------------------------------------------------------------------"
    echo "OK  : $f"
    echo "=== $(date) ==="
  } | tee "$log"
  echo
done

echo "All done. Logs in: $LOGDIR"
