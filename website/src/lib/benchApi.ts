/**
 * Typed client for the TorchCell benchmark API (base path /api/v1).
 *
 * The types below mirror the API contract. `createBenchApi` returns either the HTTP
 * client or, when the site was built with BENCH_API_MOCK=1, a client that reads the
 * fixtures in static/mock/. Nothing in this file touches `window` at import time, so
 * it is safe to import during server-side rendering; the methods themselves are only
 * called from components rendered inside <BrowserOnly>.
 */

// ---------------------------------------------------------------------------
// Contract types
// ---------------------------------------------------------------------------

/** What a dataset asks for: real values, or a score for a 0/1 label. */
export type Task = 'regression' | 'binary';

export type RegressionMetricName = 'pearson' | 'spearman' | 'mse' | 'mae' | 'r2';
export type BinaryMetricName = 'auroc' | 'auprc';
export type MetricName = RegressionMetricName | BinaryMetricName;

/** The metrics of one task: the five regression metrics, or AUROC and AUPRC. */
export type MetricSet = Partial<Record<MetricName, number>>;

export type SplitScores = {
  n_records: number;
  macro: MetricSet;
  per_target: Record<string, MetricSet>;
};

/** One source a bundle was built from, pinned by hash (see torchcell.benchmark.bundle). */
export type SourceRecord = {
  name: string;
  role: string;
  source_url: string | null;
  retrieval_method: string;
  retrieved_at: string | null;
  sha256: string;
  bytes: number | null;
  n_files: number | null;
  note: string | null;
};

export type BundleProvenance = {
  script: string;
  label_rule: string;
  split_rule: string;
  sources: SourceRecord[];
  notes: string[];
};

export type BenchmarkDatasetPublic = {
  slug: string;
  title: string;
  description: string;
  loader_class: string;
  citation_key: string;
  version: string;
  task: Task;
  targets: string[];
  n_train: number;
  n_val: number;
  n_test: number;
  primary_metric: MetricName;
  docs_url: string | null;
  tc_data_slug: string | null;
  /** Absent on bundles written before 2026-10-08 and in the mock fixtures. */
  provenance?: BundleProvenance | null;
};

export type SubmissionStatus = 'rejected' | 'provisional' | 'verified' | 'withdrawn';

/** One row of the people directory: an account with at least one scored submission. */
export type UserDirectoryEntry = UserPublic & {
  n_submissions: number;
  last_submitted_at: string;
};

export type LeaderboardRow = {
  submission_id: string;
  user_id: string;
  display_name: string;
  affiliation: string | null;
  method_name: string;
  model_family: string;
  encoding: string;
  code_url: string | null;
  status: 'provisional' | 'verified';
  submitted_at: string;
  is_baseline: boolean;
  val: SplitScores;
  test: SplitScores;
  flags: string[];
};

export type SubmissionResult = {
  submission_id: string;
  dataset_slug: string;
  status: SubmissionStatus;
  submitted_at: string;
  method_name: string;
  rejection_reasons: string[];
  val: SplitScores | null;
  test: SplitScores | null;
  flags: string[];
  archive_sha256: string | null;
};

export type Quota = {
  max_per_window: number;
  window_hours: number;
  min_gap_minutes: number;
  used_in_window: number;
  remaining: number;
  next_allowed_at: string | null;
};

export type TokenResponse = {
  access_token: string;
  token_type: 'bearer';
  expires_at: string;
};

export type UserPublic = {
  user_id: string;
  display_name: string;
  affiliation: string | null;
  /** Name of the identity provider the account signs in through (from CILogon). */
  identity_provider: string | null;
  created_at: string;
};

export type UserPrivate = UserPublic & {email: string; approved: boolean};

/** A personal API token as its owner sees it after creation; never the token itself. */
export type ApiToken = {
  token_id: string;
  name: string;
  /** The first characters of the token, enough to tell tokens apart. */
  hint: string;
  created_at: string;
  /** When the token stops working; chosen at creation, at most a year ahead. */
  expires_at: string;
  last_used_at: string | null;
};

/** Lifetimes offered when a token is created, in days; the API default is 90. */
export const TOKEN_LIFETIME_DAYS = [30, 90, 180, 365] as const;

/** A new personal API token. `token` is returned this once and is not stored. */
export type ApiTokenCreated = ApiToken & {token: string};

/** What an account may edit about itself; both fields are public. */
export type ProfileUpdate = {display_name: string; affiliation: string | null};

/**
 * Why a sign-in ended without a session. The API sends one of these to the account
 * page as `#login_error=<code>` (torchcell.benchmark.oidc.LoginError).
 */
export const LOGIN_ERRORS = {
  denied: 'The sign-in was cancelled, or the identity provider refused it.',
  failed: 'The sign-in could not be verified. Start it again from this page.',
  unavailable: 'CILogon could not be reached. Try again in a few minutes.',
  no_email:
    'That identity provider did not release an email address. Sign in with a provider that does, such as your institution.',
  idp_not_allowed: 'Accounts cannot be opened through that identity provider.',
  email_not_allowed: 'Accounts cannot be opened with that email address.',
  email_in_use:
    'An account already uses that email address through another identity provider. Sign in with the provider you used first.',
  too_many_accounts:
    'Too many new accounts were opened from this network today. Try again tomorrow.',
  disabled: 'This account is disabled.',
} as const;

export type LoginErrorCode = keyof typeof LOGIN_ERRORS;

export function isLoginErrorCode(value: string): value is LoginErrorCode {
  return Object.prototype.hasOwnProperty.call(LOGIN_ERRORS, value);
}

export type UserHistoryRow = LeaderboardRow & {dataset_slug: string};

export type UserHistory = {
  user: UserPublic;
  submissions: UserHistoryRow[];
};

/** Sent as a JSON string in the multipart field "metadata". */
export type SubmissionMetadata = {
  method_name: string;
  description: string;
  model_family: string;
  encoding: string;
  code_url: string | null;
  uses_external_data: boolean;
  external_data_description: string | null;
  hyperparameters: Record<string, string | number | boolean>;
};

export type MessageResponse = {message: string};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export type MetricInfo = {
  key: MetricName;
  label: string;
  higherIsBetter: boolean;
  task: Task;
};

/** Every metric, in display order within its task. */
export const METRICS: readonly MetricInfo[] = [
  {key: 'pearson', label: 'Pearson', higherIsBetter: true, task: 'regression'},
  {key: 'spearman', label: 'Spearman', higherIsBetter: true, task: 'regression'},
  {key: 'mse', label: 'MSE', higherIsBetter: false, task: 'regression'},
  {key: 'mae', label: 'MAE', higherIsBetter: false, task: 'regression'},
  {key: 'r2', label: 'R2', higherIsBetter: true, task: 'regression'},
  {key: 'auroc', label: 'AUROC', higherIsBetter: true, task: 'binary'},
  {key: 'auprc', label: 'AUPRC', higherIsBetter: true, task: 'binary'},
];

/** The metrics a dataset of `task` is scored with, in display order. */
export function metricsFor(task: Task): readonly MetricInfo[] {
  return METRICS.filter((m) => m.task === task);
}

/** The metrics present in a scored split, in display order. */
export function metricsIn(scores: SplitScores): readonly MetricInfo[] {
  return METRICS.filter((m) => scores.macro[m.key] !== undefined);
}

export function metricInfo(key: MetricName): MetricInfo {
  const info = METRICS.find((m) => m.key === key);
  if (!info) {
    throw new Error(`Unknown metric: ${key}`);
  }
  return info;
}

/** The label used in selectors for the macro average over targets. */
export const MACRO_TARGET = '__macro__';

/**
 * One metric value from a split: the macro average, or a single target's value.
 * Returns null when the split has no entry for that target.
 */
export function metricValue(
  scores: SplitScores,
  metric: MetricName,
  target: string = MACRO_TARGET,
): number | null {
  const set = target === MACRO_TARGET ? scores.macro : scores.per_target[target];
  return set?.[metric] ?? null;
}

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/** The API answered with a non-2xx status. */
export class BenchApiError extends Error {
  readonly status: number;
  readonly reasons: string[];
  readonly nextAllowedAt: string | null;

  constructor(status: number, message: string, reasons: string[], nextAllowedAt: string | null) {
    super(message);
    this.name = 'BenchApiError';
    this.status = status;
    this.reasons = reasons;
    this.nextAllowedAt = nextAllowedAt;
  }
}

/** The request never got an HTTP answer (API down, wrong URL, blocked by CORS). */
export class BenchApiUnreachableError extends Error {
  readonly url: string;

  constructor(url: string) {
    super(`The benchmark API is not reachable at ${url}`);
    this.name = 'BenchApiUnreachableError';
    this.url = url;
  }
}

type FastApiValidationItem = {loc?: (string | number)[]; msg?: string};

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function isSubmissionResult(value: unknown): value is SubmissionResult {
  return isRecord(value) && Array.isArray(value.rejection_reasons) && 'submission_id' in value;
}

/**
 * Turns an error body into a BenchApiError. The contract allows `{"detail": string}`
 * and `{"detail": {message, reasons?, next_allowed_at?}}`; FastAPI's own request
 * validation adds `{"detail": [{loc, msg}, ...]}`.
 */
function toApiError(status: number, body: unknown): BenchApiError {
  const detail = isRecord(body) ? body.detail : undefined;
  if (typeof detail === 'string') {
    return new BenchApiError(status, detail, [], null);
  }
  if (Array.isArray(detail)) {
    const reasons = (detail as FastApiValidationItem[]).map(
      (item) => `${(item.loc ?? []).join('.')}: ${item.msg ?? 'invalid'}`,
    );
    return new BenchApiError(status, 'The request failed validation.', reasons, null);
  }
  if (isRecord(detail)) {
    const message = typeof detail.message === 'string' ? detail.message : `HTTP ${status}`;
    const reasons = Array.isArray(detail.reasons) ? detail.reasons.map(String) : [];
    const nextAllowedAt =
      typeof detail.next_allowed_at === 'string' ? detail.next_allowed_at : null;
    return new BenchApiError(status, message, reasons, nextAllowedAt);
  }
  return new BenchApiError(status, `HTTP ${status}`, [], null);
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

export type BenchApiConfig = {
  /** API base URL including /api/v1, without a trailing slash. */
  baseUrl: string;
  /** True when the site was built with BENCH_API_MOCK=1. */
  mock: boolean;
  /** Site URL of static/mock/, with a trailing slash. Used only when `mock` is true. */
  mockBaseUrl: string;
};

export interface BenchApi {
  readonly baseUrl: string;
  readonly mock: boolean;

  health(): Promise<unknown>;

  /**
   * URL that starts a sign-in. The browser navigates to it (a plain link, not a
   * fetch); the API redirects to CILogon and, after the sign-in, back to the account
   * page with `#login_code=<code>` or `#login_error=<reason>`. Null in mock mode.
   */
  loginUrl(): string | null;
  /** Trades the one-time sign-in code for a bearer token. A code works once. */
  exchange(code: string): Promise<TokenResponse>;
  me(accessToken: string): Promise<UserPrivate>;
  updateProfile(accessToken: string, profile: ProfileUpdate): Promise<UserPrivate>;
  /** The account's personal API tokens that are not revoked, newest first. */
  tokens(accessToken: string): Promise<ApiToken[]>;
  /** Creates a personal API token for scripts. The token is in the answer only. */
  createToken(
    accessToken: string,
    name: string,
    expiresInDays: number,
  ): Promise<ApiTokenCreated>;
  revokeToken(accessToken: string, tokenId: string): Promise<MessageResponse>;

  datasets(): Promise<BenchmarkDatasetPublic[]>;
  dataset(slug: string): Promise<BenchmarkDatasetPublic>;
  /** URL of the split file (`record_id,split`), or null in mock mode. */
  splitsUrl(slug: string): string | null;
  /** URL of the empty predictions template, or null in mock mode. */
  templateUrl(slug: string): string | null;
  submissionSchema(): Promise<Record<string, unknown>>;
  /** URL of the JSON Schema for the row and metadata models, or null in mock mode. */
  submissionSchemaUrl(): string | null;

  quota(accessToken: string): Promise<Quota>;
  /**
   * Uploads predictions. Resolves with the result both when the submission was scored
   * (HTTP 201) and when it was rejected (HTTP 422 with a result-shaped body), so the
   * caller reads `status` and `rejection_reasons`. Throws BenchApiError for anything
   * else, including HTTP 429 when rate limited.
   */
  submit(
    accessToken: string,
    datasetSlug: string,
    metadata: SubmissionMetadata,
    predictions: File,
  ): Promise<SubmissionResult>;
  mine(accessToken: string): Promise<SubmissionResult[]>;
  /** Take one of the account's own scored submissions off the board; the row is kept. */
  withdraw(accessToken: string, submissionId: string, note: string): Promise<SubmissionResult>;

  leaderboard(slug: string, verifiedOnly: boolean): Promise<LeaderboardRow[]>;
  userHistory(userId: string): Promise<UserHistory>;
  /** Every account with a scored submission, most recent submitter first. */
  users(): Promise<UserDirectoryEntry[]>;
}

async function readBody(res: Response): Promise<unknown> {
  const text = await res.text();
  if (text === '') {
    return null;
  }
  const contentType = res.headers.get('content-type') ?? '';
  return contentType.includes('json') ? JSON.parse(text) : text;
}

function createHttpApi(baseUrl: string): BenchApi {
  async function send(path: string, init?: RequestInit): Promise<Response> {
    try {
      return await fetch(`${baseUrl}${path}`, init);
    } catch {
      // fetch rejects only when no HTTP response arrived.
      throw new BenchApiUnreachableError(baseUrl);
    }
  }

  async function request<T>(path: string, init?: RequestInit): Promise<T> {
    const res = await send(path, init);
    const body = await readBody(res);
    if (!res.ok) {
      throw toApiError(res.status, body);
    }
    return body as T;
  }

  const bearer = (accessToken: string): HeadersInit => ({
    Authorization: `Bearer ${accessToken}`,
  });

  const postJson = <T>(path: string, payload: unknown): Promise<T> =>
    request<T>(path, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });

  return {
    baseUrl,
    mock: false,

    health: () => request<unknown>('/health'),

    loginUrl: () => `${baseUrl}/auth/login`,
    exchange: (code) => postJson<TokenResponse>('/auth/exchange', {code}),
    me: (accessToken) => request<UserPrivate>('/auth/me', {headers: bearer(accessToken)}),
    updateProfile: (accessToken, profile) =>
      request<UserPrivate>('/auth/profile', {
        method: 'POST',
        headers: {...bearer(accessToken), 'Content-Type': 'application/json'},
        body: JSON.stringify(profile),
      }),

    tokens: (accessToken) =>
      request<ApiToken[]>('/auth/tokens', {headers: bearer(accessToken)}),
    createToken: (accessToken, name, expiresInDays) =>
      request<ApiTokenCreated>('/auth/tokens', {
        method: 'POST',
        headers: {...bearer(accessToken), 'Content-Type': 'application/json'},
        body: JSON.stringify({name, expires_in_days: expiresInDays}),
      }),
    revokeToken: (accessToken, tokenId) =>
      request<MessageResponse>(`/auth/tokens/${encodeURIComponent(tokenId)}/revoke`, {
        method: 'POST',
        headers: bearer(accessToken),
      }),

    datasets: () => request<BenchmarkDatasetPublic[]>('/datasets'),
    dataset: (slug) =>
      request<BenchmarkDatasetPublic>(`/datasets/${encodeURIComponent(slug)}`),
    splitsUrl: (slug) => `${baseUrl}/datasets/${encodeURIComponent(slug)}/splits.csv`,
    templateUrl: (slug) => `${baseUrl}/datasets/${encodeURIComponent(slug)}/template.csv`,
    submissionSchema: () => request<Record<string, unknown>>('/submission-schema'),
    submissionSchemaUrl: () => `${baseUrl}/submission-schema`,

    quota: (accessToken) => request<Quota>('/quota', {headers: bearer(accessToken)}),

    async submit(accessToken, datasetSlug, metadata, predictions) {
      const form = new FormData();
      form.append('dataset', datasetSlug);
      form.append('metadata', JSON.stringify(metadata));
      form.append('predictions', predictions, predictions.name);
      const res = await send('/submissions', {
        method: 'POST',
        headers: bearer(accessToken),
        body: form,
      });
      const body = await readBody(res);
      if (res.ok) {
        return body as SubmissionResult;
      }
      const detail = isRecord(body) ? body.detail : undefined;
      if (res.status === 422 && isSubmissionResult(detail)) {
        return detail;
      }
      throw toApiError(res.status, body);
    },

    mine: (accessToken) =>
      request<SubmissionResult[]>('/submissions/mine', {headers: bearer(accessToken)}),
    withdraw(accessToken, submissionId, note) {
      const form = new FormData();
      form.append('note', note);
      return request<SubmissionResult>(
        `/submissions/${encodeURIComponent(submissionId)}/withdraw`,
        {method: 'POST', headers: bearer(accessToken), body: form},
      );
    },
    users: () => request<UserDirectoryEntry[]>('/users'),

    leaderboard: (slug, verifiedOnly) =>
      request<LeaderboardRow[]>(
        `/leaderboard/${encodeURIComponent(slug)}?verified_only=${verifiedOnly}`,
      ),

    userHistory: (userId) =>
      request<UserHistory>(`/users/${encodeURIComponent(userId)}/submissions`),
  };
}

/**
 * Fixture-backed client for visual development (BENCH_API_MOCK=1). It sends nothing
 * to the API: reads come from static/mock/*.json, and writes return canned answers.
 */
function createMockApi(baseUrl: string, mockBaseUrl: string): BenchApi {
  async function fixture<T>(name: string): Promise<T> {
    const res = await fetch(`${mockBaseUrl}${name}.json`);
    if (!res.ok) {
      throw new BenchApiError(res.status, `No mock fixture named ${name}.json`, [], null);
    }
    return (await res.json()) as T;
  }

  return {
    baseUrl,
    mock: true,

    health: () => Promise.resolve({status: 'mock'}),

    loginUrl: () => null,
    exchange: () =>
      Promise.resolve({
        access_token: 'mock-token',
        token_type: 'bearer',
        expires_at: '9999-01-01T00:00:00Z',
      }),
    me: () => fixture<UserPrivate>('me'),
    async updateProfile(_accessToken, profile) {
      // Nothing is sent or stored: the fixture is returned with the edit applied.
      return {...(await fixture<UserPrivate>('me')), ...profile};
    },

    tokens: () => fixture<ApiToken[]>('tokens'),
    // Nothing is sent or stored: the token below is a fixed, visibly fake value.
    createToken: (_accessToken, name, expiresInDays) =>
      Promise.resolve({
        token_id: 'mock-token-new',
        name,
        hint: 'tcb_MOCKMOCK',
        created_at: new Date().toISOString(),
        expires_at: new Date(Date.now() + expiresInDays * 86_400_000).toISOString(),
        last_used_at: null,
        token: 'tcb_MOCKMOCK-not-a-real-token-nothing-was-created',
      }),
    revokeToken: () => Promise.resolve({message: 'mock: nothing was revoked'}),

    datasets: () => fixture<BenchmarkDatasetPublic[]>('datasets'),
    async dataset(slug) {
      const all = await fixture<BenchmarkDatasetPublic[]>('datasets');
      const found = all.find((d) => d.slug === slug);
      if (!found) {
        throw new BenchApiError(404, `No mock dataset named ${slug}`, [], null);
      }
      return found;
    },
    splitsUrl: () => null,
    templateUrl: () => null,
    submissionSchema: () => fixture<Record<string, unknown>>('submission-schema'),
    submissionSchemaUrl: () => null,

    quota: () => fixture<Quota>('quota'),
    submit: () => fixture<SubmissionResult>('submission-result'),
    mine: () => fixture<SubmissionResult[]>('submissions-mine'),
    async withdraw(_accessToken, submissionId) {
      const rows = await fixture<SubmissionResult[]>('submissions-mine');
      const row = rows.find((r) => r.submission_id === submissionId);
      if (!row) {
        throw new BenchApiError(404, 'unknown submission', [], null);
      }
      return {...row, status: 'withdrawn'};
    },
    users: () => fixture<UserDirectoryEntry[]>('users'),

    async leaderboard(slug, verifiedOnly) {
      const rows = await fixture<LeaderboardRow[]>(`leaderboard-${slug}`);
      return verifiedOnly ? rows.filter((r) => r.status === 'verified') : rows;
    },

    userHistory: (userId) => fixture<UserHistory>(`user-${userId}`),
  };
}

export function createBenchApi(config: BenchApiConfig): BenchApi {
  return config.mock
    ? createMockApi(config.baseUrl, config.mockBaseUrl)
    : createHttpApi(config.baseUrl);
}
