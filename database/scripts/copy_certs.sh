#!/bin/bash
# Copy renewed Let's Encrypt certificates

DOMAIN="torchcell-database.ncsa.illinois.edu"
CERT_DIR="/home/rocky/projects/torchcell/database/certificates/https"

cp -L "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$CERT_DIR/private.key"
cp -L "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$CERT_DIR/public.crt"

# The neo4j container runs as uid 67392:202 (mjvolk3, the mount owner) and reads
# the certs as that user, so they MUST be owned 67392 -- chowning to rocky:neo4j
# makes private.key (640) unreadable by the process and silently breaks TLS on the
# next renewal. See load_dump.sh / [[radiant-vm-infra]] for why 67392.
chown 67392:202 "$CERT_DIR/private.key" "$CERT_DIR/public.crt"
chmod 640 "$CERT_DIR/private.key"
chmod 644 "$CERT_DIR/public.crt"

echo "Certificates copied. Verifying new dates:"
openssl x509 -in "$CERT_DIR/public.crt" -noout -dates

echo ""
echo "Restarting Neo4j..."
docker restart tc-neo4j

# The reverse proxy (docker-compose.tc-proxy.yml) serves the same certificate straight
# from /etc/letsencrypt and loads it at start, so it must restart too or it keeps
# serving the expired one. Skipped on a host without the proxy.
if docker ps --format '{{.Names}}' | grep -qx tc-proxy; then
    echo ""
    echo "Restarting the reverse proxy..."
    docker restart tc-proxy
fi
