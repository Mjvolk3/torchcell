#!/usr/bin/env bash
# database/scripts/kg_content_hashes.sh
# [[database.scripts.kg_content_hashes]]
# https://github.com/Mjvolk3/torchcell/tree/main/database/scripts/kg_content_hashes.sh
#
# Per-dataset content hashes of a served store, streamed through cypher-shell so the
# build node never holds a dataset's ids in memory:
#
#   kg_content_hashes.sh <container> <database> <output.json> [<Dataset.id> ...]
#
# For each dataset (all Dataset nodes when none are named) the experiment ids are
# returned sorted by Neo4j and piped, one per line, into sha256sum. That is
# torchcell.knowledge_graphs.releases.content_sha256 exactly: sha256 of the sorted ids,
# newline-joined, trailing newline. Ids are hex sha256 strings, so Neo4j's ORDER BY and
# Python's sorted() agree. Output is {"<Dataset.id>": "<sha256>", ...}.
#
# Env: NEO4J_USER (neo4j), NEO4J_PASSWORD (torchcell).
set -euo pipefail

CONTAINER="${1:?container}"
DATABASE="${2:?database}"
OUTPUT="${3:?output json}"
shift 3
CS="cypher-shell -u ${NEO4J_USER:-neo4j} -p ${NEO4J_PASSWORD:-torchcell}"

if [ "$#" -gt 0 ]; then
    NAMES=("$@")
else
    mapfile -t NAMES < <(docker exec "$CONTAINER" $CS -d "$DATABASE" --format plain \
        "MATCH (d:Dataset) RETURN d.id ORDER BY d.id;" | tail -n +2 | tr -d '"')
fi

{
    echo "{"
    n=${#NAMES[@]}; i=0
    for NAME in "${NAMES[@]}"; do
        i=$((i + 1))
        HASH=$(docker exec "$CONTAINER" $CS -d "$DATABASE" --format plain \
            "MATCH (d:Dataset {id:'$NAME'})<-[:ExperimentMemberOf]-(e:Experiment) RETURN e.id ORDER BY e.id;" \
            | tail -n +2 | tr -d '"' | sha256sum | cut -d' ' -f1)
        sep=","; [ "$i" -eq "$n" ] && sep=""
        printf '  "%s": "%s"%s\n' "$NAME" "$HASH" "$sep"
        echo "  $NAME $HASH" >&2
    done
    echo "}"
} > "$OUTPUT"
echo "$n content hashes -> $OUTPUT"
