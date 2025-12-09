#!/bin/bash
# Copy renewed Let's Encrypt certificates

DOMAIN="torchcell-database.ncsa.illinois.edu"
CERT_DIR="/home/rocky/projects/torchcell/database/certificates/https"

cp -L "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$CERT_DIR/private.key"
cp -L "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$CERT_DIR/public.crt"

chown rocky:neo4j "$CERT_DIR/private.key" "$CERT_DIR/public.crt"
chmod 640 "$CERT_DIR/private.key"
chmod 644 "$CERT_DIR/public.crt"

echo "Certificates copied. Verifying new dates:"
openssl x509 -in "$CERT_DIR/public.crt" -noout -dates

echo ""
echo "Restarting Neo4j..."
docker restart tc-neo4j
