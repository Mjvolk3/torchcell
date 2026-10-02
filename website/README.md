# TorchCell website

The public website for TorchCell, built with Docusaurus 3 (classic preset,
TypeScript). It is separate from the Sphinx API docs in `docs/`, which stay where they
are; the Docs tab of this site links to them.

The final hosting location is undecided. The defaults below place the site under
`https://mjvolk3.github.io/torchcell/site/`, and both parts are configurable.

## Layout

| Path | Content |
| :-- | :-- |
| `docusaurus.config.ts` | Site config, navbar tabs, environment variables |
| `sidebars.ts` | One collapsible sidebar per tab |
| `docs/<tab>/` | The pages of each tab (MDX). Docs are served from the site root, so `docs/benchmark/metrics.mdx` is `/benchmark/metrics/` |
| `docs/benchmark/{leaderboard,submit,account,user}.mdx` | Thin pages that mount the interactive benchmark apps, so they keep the Benchmark sidebar |
| `src/components/bench/` | The interactive benchmark apps (React, browser-only) |
| `src/lib/benchApi.ts` | Typed client for the benchmark API |
| `src/components/DatasetCard.tsx` | The education card; template in `docs/education/cards/_template.mdx` |
| `src/theme/` | `MDXComponents` (components usable in MDX without an import) and `Root` (mock-data bar) |
| `static/mock/` | Fixtures for mock mode, written by `scripts/gen_mock_fixtures.mjs` |

## Build

Node 20 or later.

```bash
cd website
npm ci
npm run build       # static site in website/build
npm run typecheck   # tsc, no output on success
```

`node_modules/`, `build/`, and `.docusaurus/` are build products and are not
committed.

### Do not run a dev server on the shared database host

`npm start` (and `npx docusaurus serve`) bind a local port and keep a process
running. Do not run either on the shared host that serves the production Neo4j
database. On that host, verify with `npm run build` and `npm run typecheck` only, and
preview on your own machine.

## Environment variables

All five are read at build time by `docusaurus.config.ts`. A change needs a rebuild.

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `SITE_URL` | `https://mjvolk3.github.io` | Origin the site is served from, without a path |
| `BASE_URL` | `/torchcell/site/` | Path under that origin. Must start and end with `/` |
| `BENCH_API_URL` | `http://127.0.0.1:8725/api/v1` | Base URL of the benchmark API, including `/api/v1` |
| `BENCH_API_MOCK` | unset | `1` makes the benchmark pages read `static/mock/*.json` instead of the API |
| `ONTOLOGY_EXPLORER_URL` | `https://mjvolk3.github.io/torchcell/ontology/` | The schema explorer the Ontology tab embeds and the dataset cards link to. An absolute URL, or a path starting with `/` for a copy this site serves (render one with `python paper/nature-biotech/scripts/generate_ontology_diagram.py --explorer-only website/static/ontology-explorer/index.html` from the repo root) |

```bash
SITE_URL=https://example.org BASE_URL=/ BENCH_API_URL=https://example.org/api/v1 npm run build
```

Notes on `BENCH_API_URL`:

- The browser calls the API directly, so the API must allow the site's origin (CORS).
- A site served over HTTPS cannot call an `http://` API (mixed content). The default
  only works for a locally served site next to a locally running API.
- When the API does not answer, the benchmark pages show "The benchmark API is not
  reachable at `<url>`" and no rows.

## Mock mode

```bash
BENCH_API_MOCK=1 npm run build
```

Mock mode is for visual development only. The benchmark pages then load the fixtures
in `static/mock/` and send nothing to the API, and every page under `/benchmark/`
carries a "MOCK DATA, not real results" bar. Dataset slugs, method names, users,
dates, and numbers in the fixtures are fake by construction. The default is off; a
published build must not set `BENCH_API_MOCK`.

The fixtures are generated, not hand-written:

```bash
npm run gen-mock   # rewrites static/mock/*.json, deterministic
```

## Benchmark pages

| Page | Path | Function |
| :-- | :-- | :-- |
| Leaderboard | `/benchmark/leaderboard/` | Board and charts per dataset. `?dataset=<slug>` preselects one |
| Submit | `/benchmark/submit/` | Quota, metadata form, CSV upload, scores or rejection reasons |
| Account | `/benchmark/account/` | Sign up, log in, log out, own submissions. `?verify=<token>` confirms an email |
| User history | `/benchmark/user/?id=<user_id>` | Public record of one user's scored submissions |

The access token is kept in `sessionStorage` and sent as `Authorization: Bearer <token>`.

The site sets `trailingSlash: true`, so each page is written as
`<path>/index.html` and its canonical URL ends in `/`. A link without the trailing
slash, such as `/benchmark/account?verify=<token>`, depends on the host redirecting
to the slashed path and keeping the query string.

## Writing rules

American spelling. No em-dashes; use a comma or ` -- `. Define a term before using
it. Mark anything not built as "planned" (`<StatusBadge status="planned" />`). Do not
state an experimental detail that has not been sourced; write `<Todo />`, which
renders `TODO(source from SI)`.
