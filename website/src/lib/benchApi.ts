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

export type MetricSet = {
  pearson: number;
  spearman: number;
  mse: number;
  mae: number;
  r2: number;
};

export type MetricName = keyof MetricSet;

export type SplitScores = {
  n_records: number;
  macro: MetricSet;
  per_target: Record<string, MetricSet>;
};

export type BenchmarkDatasetPublic = {
  slug: string;
  title: string;
  description: string;
  loader_class: string;
  citation_key: string;
  version: string;
  task: 'regression';
  targets: string[];
  n_train: number;
  n_val: number;
  n_test: number;
  primary_metric: MetricName;
  docs_url: string | null;
  tc_data_slug: string | null;
};

export type SubmissionStatus = 'rejected' | 'provisional' | 'verified' | 'withdrawn';

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
  created_at: string;
};

export type UserPrivate = UserPublic & {email: string; email_verified: boolean};

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

export type SignupRequest = {
  email: string;
  password: string;
  display_name: string;
  affiliation?: string;
};

export type LoginRequest = {email: string; password: string};

export type MessageResponse = {message: string};

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

export type MetricInfo = {key: MetricName; label: string; higherIsBetter: boolean};

/** The five metrics, in display order. Pearson is the default selection. */
export const METRICS: readonly MetricInfo[] = [
  {key: 'pearson', label: 'Pearson', higherIsBetter: true},
  {key: 'spearman', label: 'Spearman', higherIsBetter: true},
  {key: 'mse', label: 'MSE', higherIsBetter: false},
  {key: 'mae', label: 'MAE', higherIsBetter: false},
  {key: 'r2', label: 'R2', higherIsBetter: true},
];

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
  if (target === MACRO_TARGET) {
    return scores.macro[metric];
  }
  const perTarget = scores.per_target[target];
  return perTarget ? perTarget[metric] : null;
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

  signup(body: SignupRequest): Promise<MessageResponse>;
  verify(token: string): Promise<MessageResponse>;
  login(body: LoginRequest): Promise<TokenResponse>;
  me(accessToken: string): Promise<UserPrivate>;

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

  leaderboard(slug: string, verifiedOnly: boolean): Promise<LeaderboardRow[]>;
  userHistory(userId: string): Promise<UserHistory>;
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

    signup: (body) => postJson<MessageResponse>('/auth/signup', body),
    verify: (token) => postJson<MessageResponse>('/auth/verify', {token}),
    login: (body) => postJson<TokenResponse>('/auth/login', body),
    me: (accessToken) => request<UserPrivate>('/auth/me', {headers: bearer(accessToken)}),

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

  const notSent: MessageResponse = {message: 'Mock mode: no request was sent.'};

  return {
    baseUrl,
    mock: true,

    health: () => Promise.resolve({status: 'mock'}),

    signup: () => Promise.resolve(notSent),
    verify: () => Promise.resolve(notSent),
    login: () =>
      Promise.resolve({
        access_token: 'mock-token',
        token_type: 'bearer',
        expires_at: '9999-01-01T00:00:00Z',
      }),
    me: () => fixture<UserPrivate>('me'),

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
