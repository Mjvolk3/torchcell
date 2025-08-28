#!/bin/bash

# Let's Encrypt setup for Neo4j on Rocky Linux
# This script sets up certbot and generates/renews certificates

PROJECT_DIR="/home/rocky/projects/torchcell"
CERT_DIR="$PROJECT_DIR/database/certificates/https"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Let's Encrypt Certificate Setup for Neo4j${NC}"
echo "========================================="

# Check if running as root/sudo
if [[ $EUID -ne 0 ]]; then
   echo -e "${RED}This script must be run with sudo for certbot installation${NC}" 
   exit 1
fi

# Install certbot on Rocky Linux
echo -e "${YELLOW}Installing certbot...${NC}"
dnf install -y epel-release
dnf install -y certbot

# Get domain name from user
echo ""
read -p "Enter your domain name (e.g., neo4j.example.com): " DOMAIN

if [[ -z "$DOMAIN" ]]; then
    echo -e "${RED}Domain name is required${NC}"
    exit 1
fi

# Get email for Let's Encrypt registration
read -p "Enter email for Let's Encrypt notifications: " EMAIL

if [[ -z "$EMAIL" ]]; then
    echo -e "${RED}Email is required for Let's Encrypt${NC}"
    exit 1
fi

# Check if port 80 is available (needed for HTTP challenge)
if lsof -Pi :80 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}Port 80 is in use. Stopping any services using it temporarily...${NC}"
    read -p "Continue? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create certificate directory
mkdir -p "$CERT_DIR"

# Generate Let's Encrypt certificate
echo -e "${YELLOW}Generating Let's Encrypt certificate...${NC}"
certbot certonly \
    --standalone \
    --non-interactive \
    --agree-tos \
    --email "$EMAIL" \
    --domains "$DOMAIN" \
    --keep-until-expiring

if [ $? -ne 0 ]; then
    echo -e "${RED}Certificate generation failed${NC}"
    echo "Make sure:"
    echo "  1. Port 80 is accessible from the internet"
    echo "  2. Domain $DOMAIN points to this server's IP"
    echo "  3. No firewall blocking port 80"
    exit 1
fi

# Copy certificates (Docker can't follow symlinks outside mounted volumes)
echo -e "${YELLOW}Copying certificates for Docker access...${NC}"
cp -L "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$CERT_DIR/private.key"
cp -L "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$CERT_DIR/public.crt"

# Set proper ownership for neo4j group access
chown -R rocky:neo4j "$CERT_DIR"
chmod 755 "$CERT_DIR"
chmod 644 "$CERT_DIR/public.crt"
chmod 640 "$CERT_DIR/private.key"

# Make sure neo4j group can read Let's Encrypt directories
chmod 755 /etc/letsencrypt/live
chmod 755 /etc/letsencrypt/archive

echo -e "${GREEN}Certificate setup complete!${NC}"
echo ""
echo "Certificate locations:"
echo "  Private key: $CERT_DIR/private.key -> /etc/letsencrypt/live/$DOMAIN/privkey.pem"
echo "  Certificate: $CERT_DIR/public.crt -> /etc/letsencrypt/live/$DOMAIN/fullchain.pem"
echo ""
echo "Certificates will expire in 90 days. Setting up auto-renewal..."

# Setup auto-renewal with cron
echo -e "${YELLOW}Setting up automatic renewal...${NC}"

# Create renewal hook script
cat > /etc/letsencrypt/renewal-hooks/deploy/neo4j-reload.sh << 'EOF'
#!/bin/bash
# Reload Neo4j after certificate renewal

DOMAIN="torchcell-database.ncsa.illinois.edu"
CERT_DIR="/home/rocky/projects/torchcell/database/certificates/https"

# Fix permissions after renewal
chmod 755 /etc/letsencrypt/live
chmod 755 /etc/letsencrypt/archive

# Copy renewed certificates for Docker access
cp -L "/etc/letsencrypt/live/$DOMAIN/privkey.pem" "$CERT_DIR/private.key"
cp -L "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" "$CERT_DIR/public.crt"
chown rocky:neo4j "$CERT_DIR/private.key" "$CERT_DIR/public.crt"
chmod 640 "$CERT_DIR/private.key"
chmod 644 "$CERT_DIR/public.crt"

# If Neo4j container is running, restart it to load new certificates
if docker ps | grep -q tc-neo4j; then
    echo "Restarting Neo4j container to load new certificates..."
    docker restart tc-neo4j
fi
EOF

chmod +x /etc/letsencrypt/renewal-hooks/deploy/neo4j-reload.sh

# Add cron job for renewal (runs twice daily as recommended)
(crontab -l 2>/dev/null | grep -v "certbot renew"; echo "0 0,12 * * * /usr/bin/certbot renew --quiet") | crontab -

echo -e "${GREEN}Auto-renewal configured!${NC}"
echo ""
echo "Next steps:"
echo "1. Update Neo4j to use HTTPS on port 7473"
echo "2. Run: sudo bash $PROJECT_DIR/database/build/build_openstack.sh"
echo "3. Access Neo4j at: https://$DOMAIN:7473"
echo ""
echo -e "${YELLOW}Note: Make sure ports 7473 and 7687 are open in your firewall${NC}"