---
id: xno7016j8rn43tb5cb6cbh3
title: Overview
desc: ''
updated: 1720423056267
created: 1720324788265
---
## 2024.07.06 - Building on GilaHyper Workstation

- 1. high cpu, high ram `biocypher-out` runs.
- 2. hosting database runs.

## 2024.07.07 - From Scratch Build Steps

```python
python -m torchcell.database.directory_setup
```

1. Create https certification.

```bash
(torchcell) michaelvolk@gilahyper torchcell % sudo certbot certonly --standalone -d gilahyper.zapto.org                                         23:32
[sudo] password for michaelvolk: 
Saving debug log to /var/log/letsencrypt/letsencrypt.log
Requesting a certificate for gilahyper.zapto.org

Successfully received certificate.
Certificate is saved at: /etc/letsencrypt/live/gilahyper.zapto.org/fullchain.pem
Key is saved at:         /etc/letsencrypt/live/gilahyper.zapto.org/privkey.pem
This certificate expires on 2024-10-06.
These files will be updated when the certificate renews.
Certbot has set up a scheduled task to automatically renew this certificate in the background.

- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
If you like Certbot, please consider supporting our work by:
 * Donating to ISRG / Let's Encrypt:   https://letsencrypt.org/donate
 * Donating to EFF:                    https://eff.org/donate-le
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
```

2. Set up the necessary directory structure.

3. Run slurm script.

We have abandoned this for the time being since we are going to stick with `http` unless it becomes an issue upon review.

## 2024.07.11 - From Scratch Build Steps (HTTP)

1.

```python
python -m torchcell.database.directory_setup
```
