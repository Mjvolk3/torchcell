---
id: p7byvrm4m2lfln3cjd1yyg9
title: Website and benchmark platform 2026.10.01
desc: 'Public website with tabs and a collapsible sidebar, and the tc-bench service: accounts, prediction submissions, a pydantic-validated grader, quota, integrity flags, and a provisional or verified leaderboard on PostgreSQL'
updated: 1790895906254
created: 1790895906254
---

## Context

The public face of the project is a Sphinx site on GitHub Pages
(`https://mjvolk3.github.io/torchcell/`): a guide, the database page, two dataset pages
and the API reference. It has no benchmark, no tutorials section, no education pages, and
nothing interactive, and a static Sphinx build cannot host accounts or submissions.

Request (2026-10-01, dictated): a modern website with tabs and a collapsible left sidebar;
tabs for overview, database, ontology visualization, docs, public benchmark, milestones,
tutorials, education, and simplified database queries. The benchmark is the core: one
board per single dataset, a standardized kNN and linear-regression baseline over gene
encodings, a downloadable test set, submissions of predictions (not scores) validated by
pydantic objects and graded by a lightweight grader, validation and test both reported,
scores marked provisional until reproduced, a toggle for reproducible versus not yet, a
login with email confirmation, one account per person as far as that can be enforced,
3 attempts per 24 hours with at least 1 hour between two, rejection with a reason when the
format is wrong, zipped storage of submissions after scoring, and later an LLM pipeline
that processes GitHub issues. ProteinGym was the reference point; its process is a pull
request plus an issue with no live grader.

This note records what was scaffolded on branch `website-benchmark-scaffold`, the
decisions behind it, how to deploy it, and what is still open. Nothing was deployed: no
container was started, no port was opened, and no service on the Radiant VM was touched.

## Relevant files

| Path | Action | Purpose |
|---|---|---|
| `torchcell/benchmark/submission.py` | NEW | the submission contract: `PredictionRow`, `SubmissionMetadata`, the JSON Schema |
| `torchcell/benchmark/bundle.py` | NEW | one benchmark dataset on disk: public `splits.csv` and `template.csv`, grader-only `labels.csv`, sha256-pinned in `benchmark.json`; `write_bundle` and a CLI |
| `torchcell/benchmark/validation.py` | NEW | CSV validation against the public template with explicit reasons; a CLI submitters run locally |
| `torchcell/benchmark/grading.py` | NEW | Pearson, Spearman, MSE, MAE, R2 per target and macro-averaged (numpy only) |
| `torchcell/benchmark/integrity.py` | NEW | the two flags on the validation and test pair |
| `torchcell/benchmark/ratelimit.py` | NEW | the quota: 3 per rolling 24 h, 1 h apart |
| `torchcell/benchmark/security.py` | NEW | argon2id passwords, JWT sessions, one-time tokens, canonical email, domain policy |
| `torchcell/benchmark/mailer.py` | NEW | SMTP over STARTTLS, console sender for development, memory sender for tests |
| `torchcell/benchmark/db.py` | NEW | SQLAlchemy schema: `users`, `email_tokens`, `submissions` |
| `torchcell/benchmark/storage.py` | NEW | deterministic zip per scored submission |
| `torchcell/benchmark/app.py` | NEW | the FastAPI service `tc-bench` and its `TC_BENCH_*` configuration |
| `tests/torchcell/benchmark/` | NEW | one test file per module plus a hand-gradable toy bundle |
| `Dockerfile.tc-bench`, `docker-compose.tc-bench.yml` | NEW | slim API image; PostgreSQL with no published port; API on loopback only |
| `docker/tc-bench/Caddyfile.example`, `docker/tc-bench/tc-bench.env.example` | NEW | reverse proxy and settings templates |
| `.github/ISSUE_TEMPLATE/benchmark-verification.yml` | NEW | the form a submitter files to have a provisional score reproduced |
| `website/` | NEW | Docusaurus site: nine tabs, collapsible sidebars, leaderboard, submit, account and per-user pages |
| `env/requirements.txt`, `pyproject.toml`, `.gitignore` | MODIFY | service dependencies, the `tc-bench-server` script, ignore rules |

## Key design decisions

1. **Two parts: a static site and one small service.** The site is static (Docusaurus) and
   can be served from GitHub Pages or from the reverse proxy. Everything that needs state
   (accounts, submissions, the board) is the `tc-bench` API. The Sphinx site stays the API
   reference and the Docs tab links to it; nothing in it is duplicated or moved.
2. **Submit predictions, never scores.** A submission is a CSV in long format with the
   columns `record_id,split,target,prediction`, one row per (record, target) pair, covering
   every validation and test pair of the dataset's public `template.csv`, plus a metadata
   JSON. The server computes every metric. Long format handles a single-target dataset and
   a multi-target one (amino acids) with the same four columns.
3. **The contract is pydantic and public.** `PredictionRow` and `SubmissionMetadata` are the
   models the server validates with, the JSON Schema at `/submission-schema` is generated
   from them, and the validator needs only the public template, so
   `python -m torchcell.benchmark.validation predictions.csv --template template.csv` gives
   a submitter the server's verdict before an attempt is spent.
4. **A wrong format is rejected with reasons, and a rejection is an attempt.** Reasons name
   the line (`line 5: v4 is in the val split, not test`); at most 20 are listed and the rest
   counted. Counting rejections is what keeps the grader from being bombarded; the local
   validator is what makes that fair.
5. **Quota: 3 attempts per rolling 24 hours and 1 hour between attempts, per account, over
   all datasets.** The check and the insert share one transaction under a lock on the user
   row, so two concurrent uploads cannot both pass. Baselines uploaded by an admin are
   outside the quota.
6. **Validation and test are always reported together, and a score is provisional until
   reproduced.** Statuses: `rejected`, `provisional`, `verified`, `withdrawn`. The board
   takes `verified_only=true`. Promotion is an admin action today; the issue form collects
   what a reproduction needs so the planned LLM pipeline has a fixed input.
7. **Cheating is flagged, not prevented.** Labels of published datasets are not secret.
   Two flags mark a submission for review and never reject it: `test_exceeds_val` (test
   better than validation by more than 0.02 on the primary metric) and
   `val_test_divergence` (over the last three scored submissions of an account on a
   dataset, test rose every time while validation did not). Both thresholds are
   uncalibrated policy defaults. Every account's scored history is public.
8. **One account per person is approximated in four layers.** A confirmed email address;
   a canonical-address unique key (`+tag` stripped, Gmail dots removed, so the usual alias
   tricks map to one account); an optional blocked-domain file and an optional
   allowed-suffix list; and an optional hold of new accounts for admin approval
   (`TC_BENCH_REQUIRE_APPROVAL=1`). None of these proves two addresses are one person.
   Hypothesis (untested): requiring an institutional suffix or approval is what actually
   stops throwaway accounts; the default leaves both off.
9. **Scored submissions are archived as a zip; rejected ones are not.** The zip holds the
   uploaded CSV, the metadata and the result, lives at
   `<submissions_root>/<slug>/<YYYY>/<MM>/<id>.zip`, is byte-deterministic, and its sha256
   is on the row. A rejected upload leaves only its sha256 and reasons, so junk cannot fill
   the disk.
10. **Baselines go through the same grader.** `POST /admin/baselines` takes a predictions
    file like any submission and stores it under a system account, so a baseline number on
    the board has the same provenance as a submitted one. The kNN and linear-regression
    runs that produce those files are not written yet.
11. **PostgreSQL is reachable only by the API, and the API only through the proxy.** The
    database publishes no port and sits on an internal Docker network. The API binds
    127.0.0.1 by default and the compose file publishes it on host loopback. The only
    public entry point is a TLS reverse proxy on 443, which is not set up.
12. **Secrets are files.** The database password, the JWT secret, the SMTP password and the
    admin key hashes are read from files, never from the image or from an environment
    value. Admin keys reuse `torchcell.api_keys` (named keys stored as sha256).

## What exists after this branch

- The service runs in tests end to end on SQLite: signup, confirmation, sign-in, lockout
  after five wrong passwords, dataset routes, scored and rejected submissions, quota,
  upload size guard, flags, verify and withdraw, approval, baselines.
- Routes under `/api/v1`: `health`, `submission-schema`, `auth/{signup,verify,login,me}`,
  `datasets`, `datasets/{slug}`, `datasets/{slug}/{splits,template}.csv`, `quota`,
  `submissions`, `submissions/mine`, `leaderboard/{slug}`, `users/{id}/submissions`, and the
  admin routes `admin/submissions/{id}/{verify,withdraw}`, `admin/users/{id}/{approve,disable}`,
  `admin/baselines`. Swagger is at `/docs`.
- The website has the nine tabs, one collapsible sidebar each. Tutorials, education cards,
  encodings and query pages are stubs marked planned; experimental details on the cards
  are `TODO(source from SI)` placeholders. The leaderboard, submit, account and per-user
  pages are React apps against the API (`website/src/components/bench/`), with a
  `BENCH_API_MOCK=1` build flag that loads clearly labeled mock fixtures for visual work.
- The site uses trailing slashes, so `TC_BENCH_ACCOUNT_URL` must end in `/`.

## Deployment runbook (not executed)

Each step that changes the shared Radiant VM is listed so it can be reviewed first.

1. Decide the host name and whether the site is served from GitHub Pages or the proxy.
2. Copy `docker/tc-bench/tc-bench.env.example` to `.env.tc-bench`, create the four secret
   files it describes, and create the datasets and submissions directories on Taiga. The
   example defaults to `/mnt/zhao5/mjvolk3/projects/torchcell/data/torchcell/tc-bench/`
   (`datasets/` and `submissions/`), beside the `tc-data` store and the raw mirror.
3. Build bundles into the datasets directory (`python -m torchcell.benchmark.bundle`).
4. `docker compose --env-file .env.tc-bench -f docker-compose.tc-bench.yml build`, mint an
   admin key with `--gen-admin-key`, start `tc-bench-db`, run `--init-db`, then `up -d`.
   This pulls `postgres:17-alpine` and `python:3.13-slim` and creates a named volume on the
   40 GB root disk, which was 85% full on 2026-10-01; check `df -h /` first.
5. Check `curl http://127.0.0.1:8725/api/v1/health` on the VM.
6. Install the reverse proxy from `docker/tc-bench/Caddyfile.example` and open TCP 443 (and
   80 for the certificate challenge) in the OpenStack security group. This is the step
   that makes anything public, and it needs the NCSA-side change.
7. Build the site with `BENCH_API_URL` set to the public API URL and publish it.

## Gotchas

1. `tests/torchcell/conftest.py` imports torch. On a machine without torch the benchmark
   tests run with `--confcutdir=tests/torchcell/benchmark`.
2. PostgreSQL must not sit on the Taiga NFS mount; the compose file uses a local named
   volume for that reason.
3. The API container runs as a non-root uid with a read-only root filesystem; the secret
   files must be readable by that uid, and uploads spool to a 128 MB tmpfs.
4. Bundles are loaded at startup; adding a dataset needs a restart.
5. `db.py` creates tables with `create_all`. There is no migration tool yet, so the first
   column change after deployment needs Alembic before it ships.
6. A session token is checked against the wall clock by PyJWT, so `login` issues it on the
   wall clock even when a test clock is injected.
7. This Radiant checkout is sparse (`.claude`, `biocypher`, `database`); the worktree for
   this branch is sparse too and leaves out `notes/assets`, `experiments`, `paper`,
   `notes-tex`, `data` and `profiles`.

## Verification

- `pytest tests/torchcell/benchmark` (Python 3.13.15, `--confcutdir` as in Gotcha 1):
  143 passed, 2026-10-01, Radiant VM.
- `ruff check` and `ruff format` clean on `torchcell/benchmark` and
  `tests/torchcell/benchmark`; the CI mypy command (`--config-file=pyproject.toml
  --follow-imports=silent` on the changed files) reports no issues in 24 files;
  `scripts/check_paired_tests.py` pairs all 11 new modules; `scripts/test_quality_check.py`
  is clean on the 11 test files.
- Website: `npm run typecheck` and `npm run build` succeed (Docusaurus 3.10.2, Node 22.18,
  broken links set to fail the build), run in a scratch copy so no `node_modules` enters
  the worktree.
- `docker compose -f docker-compose.tc-bench.yml config` parses, and renders the API port
  with `host_ip: 127.0.0.1`.
- Not run: the service against a real PostgreSQL server (the schema was compiled for the
  PostgreSQL dialect in a test, nothing more), the Docker image build, SMTP delivery, the
  reverse proxy, the full repository test suite (no torch on this machine), and the
  pre-commit hooks (not installed in this clone; markdownlint was not run on the two new
  notes). The interactive benchmark pages were never opened in a browser: their charts,
  layout, dark mode and mobile view are unchecked.

## Open questions

- **Which three datasets start the benchmark.** No note records the choice. The website
  seeds `smf-costanzo2016`, `amino-acid-mulleder2016` and `betaxanthin-cachera2023`
  (continuous targets, already on dataset pages) and marks the choice provisional.
- **Record ids and splits.** A bundle needs a stable record id per experiment and a pinned
  split. The script that exports both from a built dataset belongs in an experiment folder
  and is not written; `CellDataModule` pinned split indices are the likely source.
- **Baseline procedure.** kNN and linear regression over which encodings, with which
  hyperparameter grid selected on validation; then the runs, uploaded through
  `/admin/baselines`.
- **Hosting.** GitHub Pages already serves the Sphinx site at the repository root path;
  the Docusaurus site needs either a subpath there or the proxy on the Radiant host.
- **Account policy.** Whether to require an institutional email suffix or admin approval.
- **Verification pipeline.** The LLM-driven reproduction of a submission from its issue
  form; today `verify` is a manual admin call.
- **Password reset** and account deletion routes are not written.

## 2026.10.02 - Sign-in moved to CILogon

Accounts no longer have passwords. A person signs in at CILogon (OpenID Connect) with
their institution, or with ORCID, GitHub, Google or Microsoft, and the service keeps the
identity CILogon asserts. This supersedes the email-and-password design above: decision 8
(a confirmed email address as the first layer), the `mailer.py` and argon2 rows of the
file table, the SMTP secret in decision 12, the `auth/{signup,verify,login,me}` routes,
and the open question on password reset, which no longer applies.

### Why CILogon, and why inside the API

- iCloudBiofoundry already signs users in through CILogon, behind an `oauth2-proxy`
  service whose cookie the backend resolves to an email (`iBioFoundry/ibiofoundry-backend`,
  `backend/app/deps.py`). That cookie is scoped to its own domain, so it cannot be reused
  here, and a cookie-based proxy would force the website and the API onto one parent
  domain.
- The flow therefore runs inside `tc-bench` with Authlib's Starlette client. No container
  is added on the borrowed VM, the session stays a bearer token so the static site can be
  hosted anywhere, and the full CILogon claims are available, `idp` and `idp_name` among
  them, which the proxy's `/userinfo` does not pass on.
- Authlib performs every protocol step (discovery, `state`, `nonce`, PKCE S256, the token
  exchange, and ID token verification against CILogon's keys). The code owned here is the
  three routes and the account rules.

### What changed

| Path | Change |
|:--|:--|
| `torchcell/benchmark/oidc.py` | NEW: `OidcConfig`, `Identity`, `LoginError`, `identity_from_claims`, `build_provider` |
| `torchcell/benchmark/app.py` | `GET /auth/login`, `GET /auth/callback`, `POST /auth/exchange`, `POST /auth/profile`; signup, verify and password login removed |
| `torchcell/benchmark/db.py` | `users` keyed by `(oidc_issuer, oidc_subject)`, with `idp`, `idp_name`, `last_login_at`; no password column; `login_codes` replaces `email_tokens` |
| `torchcell/benchmark/security.py` | password hashing removed; `AccountPolicy.allowed_idps`; `derive_key` |
| `torchcell/benchmark/mailer.py` | REMOVED, with its test: nothing sends mail |
| `tests/torchcell/benchmark/_fake_idp.py` | NEW: an in-process OpenID provider with a real RSA key, used through the HTTP transport |
| `website/src/components/bench/AccountApp.tsx` | one "Sign in with CILogon" link, the code exchange, a public-profile form |
| `docker-compose.tc-bench.yml`, `Dockerfile.tc-bench`, `docker/tc-bench/*` | `cilogon_client_secret` replaces the SMTP secret; `TC_BENCH_PUBLIC_URL`, `TC_BENCH_CILOGON_CLIENT_ID` |

### Decisions

1. **The account key is `(iss, sub)`.** CILogon's `sub` is stable for one person at one
   identity provider. The email address is stored and must be present, and its canonical
   form stays unique.
2. **A second identity on the same address is refused, not merged.** Merging by email
   would let whichever provider releases an address claim the account that already holds
   it.
3. **The browser never sees the session token in a URL.** The callback redirects to the
   account page with a one-time code in the URL fragment; the page removes it from the
   address bar and posts it to `/auth/exchange`. The code is stored as its sha256, works
   once, and expires after two minutes.
4. **The sign-in state lives in a signed cookie, scoped to the sign-in routes.** It holds
   the `state`, `nonce` and PKCE verifier for at most ten minutes, is `HttpOnly`,
   `SameSite=Lax`, `Secure` when the callback is HTTPS, and is signed with a key derived
   from the session secret. A callback that arrives in a browser without that cookie is
   refused.
5. **A refused sign-in carries a code, never provider text.** `LoginError` has nine
   values; the account page maps each to a sentence.
6. **Policy levers kept from the first design:** new accounts per client address per
   24 hours, admin approval, blocked or required email domains. New:
   `TC_BENCH_ALLOWED_IDPS_FILE` restricts sign-in to listed providers.

### Verification

- `pytest tests/torchcell/benchmark`: 172 passed. Against the fake provider the tests
  cover the redirect parameters, a forged state, a callback in another browser, a
  replayed callback, an ID token with the wrong nonce, audience, issuer, expiry or
  signature, a cancelled sign-in, a provider outage at discovery and at the token
  endpoint, a missing or malformed email, the policy refusals, and code reuse and expiry.
- `ruff`, strict `mypy` (27 files), `scripts/check_paired_tests.py` and
  `scripts/test_quality_check.py` are clean.
- A browser run in headless Chromium (19 of 19 checks): a non-mock build of the site, the
  service on a loopback port with the fake provider's authorization endpoint served by
  the same process. Sign in, the address bar without a code afterwards, the token in
  `sessionStorage` only, reload, profile edit, the submit page accepting the session,
  sign out, and a refused sign-in with its reason.
- Checked against the real service: CILogon's discovery document lists PKCE `S256`,
  `RS256`, the four scopes requested, and the `idp` and `idp_name` claims.
- Not run: a sign-in against CILogon itself. That needs a registered client, which needs
  the public callback URL, which needs the hosting decision.

### Before it can go live

1. Decide the public URL of the API (`TC_BENCH_PUBLIC_URL`).
2. Register a client at <https://cilogon.org/oauth2/register> with the callback
   `<public URL>/api/v1/auth/callback` and the scopes `openid`, `email`, `profile`,
   `org.cilogon.userinfo`. CILogon approves registrations by hand.
3. Put the client id in `.env.tc-bench` and the client secret in the
   `cilogon_client_secret` file, then follow the runbook above.
4. Decide whether `TC_BENCH_ALLOWED_IDPS_FILE` restricts providers and whether
   `TC_BENCH_REQUIRE_APPROVAL` is on.

## 2026.10.04 - Personal API tokens, tc-bench client, one sidebar shape

### Submitting from a script

- A signed-in account creates personal API tokens on the account page
  (`POST /auth/tokens`, at most 5, each shown once and stored as its sha256, revocable).
  A token is `tcb_` plus 32 random bytes and is sent as `Authorization: Bearer`, the
  same header as the browser's session token; the prefix tells them apart.
- A token reads the account, reads the quota, submits and lists attempts. Editing the
  profile and managing tokens need the session token, so a leaked token cannot mint
  others. The quota stays per account.
- `torchcell/benchmark/client.py` holds `BenchClient` and the `tc-bench` command
  (`datasets`, `template`, `quota`, `submit`, `mine`). `submit` validates against the
  template before uploading, so a malformed file does not spend an attempt.
  Environment: `TC_BENCH_URL`, `TC_BENCH_TOKEN`.
- Reading needs no account: datasets, splits, templates, boards and public histories
  are open, and the client works without a token for those calls.
- `Quota` and `SubmissionResult` moved to `torchcell/benchmark/results.py` so the
  server and the client share them.
- The submit page leads with the API setup; the form is the same `POST /submissions`,
  labels show the request field names, and "This form as an API request" prints the
  form's values as `metadata.json` plus a `curl` command.

### Sidebar

- Every tab has the same shape: landing page first, then groups that are always-open
  headings (`sidebarCollapsible: false`). No dropdowns, so every clickable sidebar
  entry opens a page.

### Verified, and not

- 194 tests pass in `tests/torchcell/benchmark` (in-process app, SQLite, fake identity
  provider); ruff, strict mypy, paired-tests and test-quality gates pass on the changed
  files.
- Mock-mode preview checked in headless Chromium: sidebar on five tabs, submit page
  sections and code samples, token creation panel, form request preview; no console
  errors. `tsc --noEmit` is clean.
- Not run: the token routes against PostgreSQL, the `tc-bench` command against a
  deployed API (none exists yet), a real CILogon sign-in.
- A deployed database created before this change needs the `api_tokens` table
  (`--init-db` creates missing tables; there is still no migration tool).
