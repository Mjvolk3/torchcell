#!/bin/bash

# This script:
# 1. Updates neo4j admin password to match YAML config
# 2. Creates a read-only user that cannot elevate privileges
# Run AFTER the database is imported but BEFORE setting to read-only

echo "Setting up Neo4j users and passwords..."

# Initial password used when container was created
INITIAL_PASSWORD="torchcell"

# Get the expected password from YAML config
PRODUCTION_CONFIG="/var/lib/neo4j/biocypher/config/PRODUCTION_linux-amd_biocypher_config.yaml"
YAML_PASSWORD=$(docker exec tc-neo4j bash -c "grep '  password:' $PRODUCTION_CONFIG 2>/dev/null | awk '{print \$2}'")

if [ -z "$YAML_PASSWORD" ]; then
    echo "ERROR: Could not extract password from YAML config"
    echo "Please check $PRODUCTION_CONFIG exists and contains password field"
    exit 1
fi

echo "Found YAML password: [hidden]"

# Update neo4j password if needed
echo "Updating neo4j admin password..."
if docker exec tc-neo4j cypher-shell -u neo4j -p "$INITIAL_PASSWORD" -d system "ALTER USER neo4j SET PASSWORD '${YAML_PASSWORD}';" 2>/dev/null; then
    echo "Successfully updated neo4j password from initial to YAML config"
    ADMIN_PASSWORD="$YAML_PASSWORD"
elif docker exec tc-neo4j cypher-shell -u neo4j -p "$YAML_PASSWORD" -d system "SHOW USERS;" &>/dev/null; then
    echo "Neo4j password already matches YAML config"
    ADMIN_PASSWORD="$YAML_PASSWORD"
else
    echo "ERROR: Could not connect with either initial or YAML password"
    exit 1
fi

# Read-only user configuration
READER_PASSWORD="${1:-ReadOnly}"
ADMIN_USER="neo4j"

echo ""
echo "Creating read-only user..."

# Drop existing reader user if it exists
if docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system "SHOW USERS;" 2>/dev/null | grep -q "reader"; then
    echo "Dropping existing reader user..."
    docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system "DROP USER reader;" 2>/dev/null
fi

# Create the reader user
echo "Creating user 'reader' with password '$READER_PASSWORD'..."
if ! docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system "CREATE USER reader SET PASSWORD '${READER_PASSWORD}' SET PASSWORD CHANGE NOT REQUIRED;"; then
    echo "ERROR: Failed to create reader user"
    exit 1
fi

# Grant read-only permissions with explicit DENY to prevent elevation
echo "Setting reader permissions..."
docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system <<EOF || exit 1
// Grant built-in reader role
GRANT ROLE reader TO reader;

// Grant database access
GRANT ACCESS ON DATABASE torchcell TO reader;

// Grant read privileges
GRANT MATCH {*} ON GRAPH torchcell TO reader;

// CRITICAL: Explicitly DENY all write operations
// This prevents ANY privilege elevation
DENY CREATE ON GRAPH torchcell TO reader;
DENY DELETE ON GRAPH torchcell TO reader;
DENY SET PROPERTY ON GRAPH torchcell TO reader;
DENY REMOVE PROPERTY ON GRAPH torchcell TO reader;
DENY WRITE ON GRAPH torchcell TO reader;
DENY ALL DBMS PRIVILEGES ON DBMS TO reader;
EOF

# Verify users were created
echo ""
echo "Verifying users..."
USERS=$(docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system "SHOW USERS;" 2>/dev/null)
echo "$USERS" | grep -E "neo4j|reader" || echo "Warning: Could not verify users"

# Test reader access (database must be started first)
echo ""
echo "Testing reader access..."
# Ensure database is started before testing
docker exec tc-neo4j cypher-shell -u "$ADMIN_USER" -p "$ADMIN_PASSWORD" -d system "START DATABASE torchcell;" 2>/dev/null || true
sleep 2

if docker exec tc-neo4j cypher-shell -u reader -p "$READER_PASSWORD" -d torchcell "MATCH (n) RETURN COUNT(n) as count LIMIT 1;" &>/dev/null; then
    echo "✓ Reader can access torchcell database"
    # Note: Cannot reliably test write prevention here since database isn't READ ONLY yet
    # The database will be set to READ ONLY after this script completes
else
    echo "⚠ Warning: Reader cannot access database yet (may need to wait for database to start)"
fi

echo ""
echo "User setup complete!"
echo ""
echo "=================================================="
echo "Database Access Credentials:"
echo "=================================================="
echo "Admin access:"
echo "  Username: neo4j"
echo "  Password: [from YAML config]"
echo "  Database: torchcell"
echo ""
echo "Read-only access:"
echo "  Username: reader"
echo "  Password: $READER_PASSWORD"
echo "  Database: torchcell"
echo ""
echo "The reader user has PERMANENT read-only access."
echo "DENY permissions prevent ANY privilege elevation."
echo "=================================================="
echo ""
echo "Note: Database will be set to READ ONLY mode after this script."