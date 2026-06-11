#!/bin/bash
# Certbot deploy-hook for the TorchCell Neo4j TLS certificate.
#
# Certbot runs every executable in /etc/letsencrypt/renewal-hooks/deploy/
# exactly once per lineage that ACTUALLY renewed (not on no-op renew checks),
# as root, with an arbitrary working directory. It exports RENEWED_DOMAINS
# (space-separated) and RENEWED_LINEAGE for the renewed cert.
#
# This hook copies the renewed cert into the dir Neo4j serves and restarts the
# container by delegating to copy_certs.sh -- the single source of truth for
# that logic. Without this wiring, certbot renews the cert under
# /etc/letsencrypt/live/ but Neo4j keeps serving the stale copy until it
# expires (the failure mode that produced the browser "not secure" warning).
#
# Install (requires sudo; certbot owns /etc/letsencrypt):
#   sudo cp /home/rocky/projects/torchcell/database/scripts/letsencrypt-deploy-hook.sh \
#           /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh
#   sudo chmod +x /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh
#
# Test without forcing a real renewal (avoids Let's Encrypt rate limits):
#   sudo RENEWED_DOMAINS="torchcell-database.ncsa.illinois.edu" \
#        /etc/letsencrypt/renewal-hooks/deploy/torchcell-neo4j.sh

DOMAIN="torchcell-database.ncsa.illinois.edu"

# Act only when OUR domain is among the renewed lineages. Certbot may manage
# other certs in the future; their renewals must not touch Neo4j.
case " $RENEWED_DOMAINS " in
    *" $DOMAIN "*) ;;
    *) exit 0 ;;
esac

exec /home/rocky/projects/torchcell/database/scripts/copy_certs.sh
