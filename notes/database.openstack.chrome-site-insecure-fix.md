---
id: 500m8xjbhsdvyi7a038zyk7
title: Chrome Site Insecure Fi
desc: ''
updated: 1765239291259
created: 1765239284538
---

## Problem

Chrome showing "Not Secure" for `https://torchcell-database.ncsa.illinois.edu:7473/browser/` even though the Let's Encrypt certificate had been renewed.

## Solution

### Step 1: Renew Let's Encrypt Certificate

```bash
sudo certbot renew --force-renewal
```

### Step 2: Copy Renewed Certificates to Project Directory

The deploy hook should do this automatically, but if not:

```bash
sudo bash /home/rocky/projects/torchcell/database/scripts/copy_certs.sh
```

This script copies certs from `/etc/letsencrypt/live/torchcell-database.ncsa.illinois.edu/` to the project's certificate directory and restarts Neo4j.

### Step 3: Verify Server is Serving New Certificate

```bash
echo | openssl s_client -connect localhost:7473 -servername torchcell-database.ncsa.illinois.edu 2>/dev/null | openssl x509 -noout -dates
```

Should show the new dates (e.g., `notAfter=Mar 8 21:19:36 2026 GMT`).

### Step 4: Clear Chrome's SSL Cache

Regular cache clearing doesn't work for SSL certificates. Do this instead:

1. Go to `chrome://net-internals/#hsts`
2. Under "Delete domain security policies", enter `torchcell-database.ncsa.illinois.edu`
3. Click **Delete**

4. Go to `chrome://net-internals/#sockets`
5. Click **Flush socket pools**

6. Restart Chrome completely (quit and reopen, or use `chrome://restart`)

### Verification

Visit `https://torchcell-database.ncsa.illinois.edu:7473/browser/` - should now show secure connection with valid certificate.
