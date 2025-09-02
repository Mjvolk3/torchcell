#!/bin/bash

# Generate SSL certificates for Neo4j HTTPS
# This creates self-signed certificates for development/testing

PROJECT_DIR="/home/rocky/projects/torchcell"
CERT_DIR="$PROJECT_DIR/database/certificates/https"

echo "Creating certificate directory structure..."
mkdir -p "$CERT_DIR"

# Generate private key
echo "Generating private key..."
openssl genrsa -out "$CERT_DIR/private.key" 2048

# Generate certificate signing request
echo "Generating certificate signing request..."
openssl req -new -key "$CERT_DIR/private.key" -out "$CERT_DIR/request.csr" \
    -subj "/C=US/ST=State/L=City/O=TorchCell/CN=localhost"

# Generate self-signed certificate (valid for 365 days)
echo "Generating self-signed certificate..."
openssl x509 -req -days 365 -in "$CERT_DIR/request.csr" \
    -signkey "$CERT_DIR/private.key" -out "$CERT_DIR/public.crt"

# Clean up CSR (not needed after certificate is generated)
rm "$CERT_DIR/request.csr"

# Set appropriate permissions
chmod 600 "$CERT_DIR/private.key"
chmod 644 "$CERT_DIR/public.crt"

# Set ownership to rocky:neo4j for consistency
chown -R rocky:neo4j "$CERT_DIR"

echo "SSL certificates generated successfully!"
echo "  Private key: $CERT_DIR/private.key"
echo "  Certificate: $CERT_DIR/public.crt"
echo ""
echo "Note: This is a self-signed certificate for development."
echo "For production, use certificates from a trusted CA."