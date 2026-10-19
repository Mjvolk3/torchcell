#!/bin/bash
# experiments/044-organism-similarity/scripts/fetch_eggnog_db.sh
# [[experiments.044-organism-similarity.scripts.fetch_eggnog_db]]
#
# Fetch the eggNOG-mapper 5.0.2 database into $DATA_ROOT/data/eggnog, once.
#
# `download_eggnog_data.py -y --data_dir ...` (emapper 2.1.12) hard-codes the host
# eggnogdb.embl.de, which no longer resolves (2026-10-19: "unable to resolve host
# address"); the same emapperdb-5.0.2 files are served from eggnog5.embl.de. This runs
# the three wget + decompress steps that script would run, against that host, and
# records the sha256 of each downloaded archive in eggnog_db_provenance.tsv.
#
#   bash experiments/044-organism-similarity/scripts/fetch_eggnog_db.sh
set -euo pipefail
set -a; source "$(git rev-parse --show-toplevel)/.env"; set +a
DB="$DATA_ROOT/data/eggnog"
BASE="http://eggnog5.embl.de/download/emapperdb-5.0.2"
mkdir -p "$DB"
cd "$DB"
PROV="$DB/eggnog_db_provenance.tsv"
printf 'file\tsource_url\tsha256\tretrieved_at\n' > "$PROV"
for f in eggnog.db.gz eggnog.taxa.tar.gz eggnog_proteins.dmnd.gz; do
  if [ ! -s "$f" ]; then
    wget -q --user-agent=Mozilla/5.0 -O "$f" "$BASE/$f"
  fi
  printf '%s\t%s\t%s\t%s\n' "$f" "$BASE/$f" "$(sha256sum "$f" | cut -d' ' -f1)" \
    "$(date -u -r "$f" +%Y-%m-%dT%H:%M:%SZ)" >> "$PROV"
done
gunzip -k eggnog.db.gz
tar -zxf eggnog.taxa.tar.gz
gunzip -k eggnog_proteins.dmnd.gz
echo "done: $(ls -la "$DB")"
