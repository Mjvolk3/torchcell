// Generates the fixtures in static/mock/ that the benchmark pages read when the site
// is built with BENCH_API_MOCK=1. Run from website/ with `npm run gen-mock`.
//
// Everything written here is fake by construction: dataset slugs, method names, user
// names, dates, and every number come from the seeded generator below, not from any
// TorchCell dataset or model. The fixtures exist only so the pages can be laid out
// without a running API. The output is deterministic, so rerunning it changes nothing.

import {mkdirSync, writeFileSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const OUT_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', 'static', 'mock');

// mulberry32: a small seeded PRNG, so the fixtures are reproducible.
function prng(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const round = (x) => Math.round(x * 10000) / 10000;

function metricSet(rand, quality) {
  const pearson = Math.min(0.99, Math.max(0.01, quality + (rand() - 0.5) * 0.04));
  return {
    pearson: round(pearson),
    spearman: round(pearson - 0.02 * rand()),
    mse: round(1 - pearson * pearson + 0.05 * rand()),
    mae: round(Math.sqrt(1 - pearson * pearson) * 0.8),
    r2: round(pearson * pearson - 0.03 * rand()),
  };
}

function macroOf(sets) {
  const keys = ['pearson', 'spearman', 'mse', 'mae', 'r2'];
  const out = {};
  for (const key of keys) {
    out[key] = round(sets.reduce((sum, s) => sum + s[key], 0) / sets.length);
  }
  return out;
}

function splitScores(rand, targets, nRecords, quality) {
  const perTarget = {};
  for (const target of targets) {
    perTarget[target] = metricSet(rand, quality);
  }
  return {n_records: nRecords, macro: macroOf(Object.values(perTarget)), per_target: perTarget};
}

const DATASETS = [
  {
    slug: 'mock-single-target',
    title: '[mock] Single-target dataset',
    description: 'Mock fixture with one target. Not a real dataset.',
    loader_class: 'MockSingleTargetDataset',
    citation_key: 'mock0000',
    version: 'mock',
    task: 'regression',
    targets: ['mock_target'],
    n_train: 80,
    n_val: 10,
    n_test: 10,
    primary_metric: 'pearson',
    docs_url: null,
    tc_data_slug: null,
  },
  {
    slug: 'mock-multi-target',
    title: '[mock] Multi-target dataset',
    description: 'Mock fixture with three targets. Not a real dataset.',
    loader_class: 'MockMultiTargetDataset',
    citation_key: 'mock0001',
    version: 'mock',
    task: 'regression',
    targets: ['mock_target_1', 'mock_target_2', 'mock_target_3'],
    n_train: 80,
    n_val: 10,
    n_test: 10,
    primary_metric: 'pearson',
    docs_url: null,
    tc_data_slug: null,
  },
];

const USERS = [
  {
    user_id: 'mock-user-a',
    display_name: 'Mock User A',
    affiliation: 'Mock Institute',
    identity_provider: 'Mock University',
  },
  {
    user_id: 'mock-user-b',
    display_name: 'Mock User B',
    affiliation: null,
    identity_provider: 'Mock University',
  },
  {
    user_id: 'mock-baselines',
    display_name: 'Mock Baselines',
    affiliation: null,
    identity_provider: null,
  },
];

// One entry per mock submission: [user index, method, family, encoding, status,
// baseline, day offset, validation quality, test minus validation, flags].
const SUBMISSIONS = [
  [2, 'mock-baseline-knn', 'kNN', 'mock-encoding-1', 'verified', true, 0, 0.3, -0.02, []],
  [2, 'mock-baseline-linear', 'linear', 'mock-encoding-1', 'verified', true, 0, 0.36, -0.03, []],
  [0, 'mock-method-a', 'mock-family-x', 'mock-encoding-2', 'verified', false, 3, 0.42, -0.02, []],
  [1, 'mock-method-b', 'mock-family-y', 'mock-encoding-1', 'provisional', false, 6, 0.47, -0.04, []],
  [0, 'mock-method-c', 'mock-family-x', 'mock-encoding-3', 'provisional', false, 11, 0.51, -0.01, []],
  [
    1,
    'mock-method-d',
    'mock-family-z',
    'mock-encoding-2',
    'provisional',
    false,
    15,
    0.44,
    0.09,
    ['mock-flag-test-exceeds-validation'],
  ],
  [0, 'mock-method-e', 'mock-family-x', 'mock-encoding-3', 'verified', false, 20, 0.56, -0.03, []],
];

function isoDay(offset) {
  // 2000-01-01 plus the offset: dates that cannot be mistaken for real submissions.
  return new Date(Date.UTC(2000, 0, 1 + offset, 12, 0, 0)).toISOString().replace('.000Z', 'Z');
}

function leaderboardFor(dataset, seed) {
  const rand = prng(seed);
  return SUBMISSIONS.map(
    ([userIndex, method, family, encoding, status, baseline, day, quality, gap, flags], i) => {
      const user = USERS[userIndex];
      return {
        submission_id: `${dataset.slug}-${String(i + 1).padStart(3, "0")}`,
        user_id: user.user_id,
        display_name: user.display_name,
        affiliation: user.affiliation,
        method_name: method,
        model_family: family,
        encoding,
        code_url: baseline || i % 2 === 0 ? 'https://example.invalid/mock-code' : null,
        status,
        submitted_at: isoDay(day),
        is_baseline: baseline,
        val: splitScores(rand, dataset.targets, dataset.n_val, quality),
        test: splitScores(rand, dataset.targets, dataset.n_test, quality + gap),
        flags,
      };
    },
  );
}

function write(name, value) {
  writeFileSync(join(OUT_DIR, `${name}.json`), `${JSON.stringify(value, null, 2)}\n`);
}

mkdirSync(OUT_DIR, {recursive: true});

write('datasets', DATASETS);

const boards = new Map();
DATASETS.forEach((dataset, index) => {
  const rows = leaderboardFor(dataset, 1000 + index);
  boards.set(dataset.slug, rows);
  write(`leaderboard-${dataset.slug}`, rows);
});

for (const user of USERS) {
  const submissions = [];
  for (const [slug, rows] of boards) {
    for (const row of rows.filter((r) => r.user_id === user.user_id)) {
      submissions.push({...row, dataset_slug: slug});
    }
  }
  write(`user-${user.user_id}`, {
    user: {...user, created_at: isoDay(0)},
    submissions,
  });
}

write('me', {
  ...USERS[0],
  created_at: isoDay(0),
  email: 'mock-user-a@example.invalid',
  approved: true,
});

write('tokens', [
  {
    token_id: 'mock-token-001',
    name: 'mock laptop',
    hint: 'tcb_MOCKaaaa',
    created_at: isoDay(3),
    expires_at: isoDay(93),
    last_used_at: isoDay(18),
  },
  {
    token_id: 'mock-token-002',
    name: 'mock cluster',
    hint: 'tcb_MOCKbbbb',
    created_at: isoDay(1),
    expires_at: isoDay(366),
    last_used_at: null,
  },
]);

write('quota', {
  max_per_window: 3,
  window_hours: 24,
  min_gap_minutes: 60,
  used_in_window: 1,
  remaining: 2,
  next_allowed_at: null,
});

const firstBoard = boards.get(DATASETS[0].slug);
const scored = firstBoard.find((r) => r.method_name === 'mock-method-e');
const scoredResult = {
  submission_id: scored.submission_id,
  dataset_slug: DATASETS[0].slug,
  status: 'provisional',
  submitted_at: scored.submitted_at,
  method_name: scored.method_name,
  rejection_reasons: [],
  val: scored.val,
  test: scored.test,
  flags: [],
  archive_sha256: '0'.repeat(64),
};
write('submission-result', scoredResult);

write('submissions-mine', [
  scoredResult,
  {
    submission_id: 'mock-rejected-001',
    dataset_slug: DATASETS[0].slug,
    status: 'rejected',
    submitted_at: isoDay(18),
    method_name: 'mock-method-rejected',
    rejection_reasons: [
      'mock reason: 3 test records have no prediction row',
      'mock reason: column "prediction" holds a non-numeric value on line 12',
    ],
    val: null,
    test: null,
    flags: [],
    archive_sha256: null,
  },
]);

write('submission-schema', {
  mock: true,
  note: 'Mock fixture. The real JSON Schema is served by GET /submission-schema.',
});

console.log(`Wrote mock fixtures to ${OUT_DIR}`);
