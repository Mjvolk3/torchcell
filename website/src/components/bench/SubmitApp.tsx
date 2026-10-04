import React, {useState, type FormEvent, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {
  metricsIn,
  type Quota,
  type SplitScores,
  type SubmissionMetadata,
  type SubmissionResult,
} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useBenchSession} from '@site/src/lib/benchAuth';
import {failureState, useLoad, type LoadState} from '@site/src/lib/useLoad';
import CodeBlock from '@theme/CodeBlock';
import StatusBadge from '@site/src/components/StatusBadge';
import {
  ApiUnreachable,
  BenchFrame,
  ErrorNotice,
  FlagChips,
  LoadGate,
  fmtDateTime,
  fmtMetric,
} from './common';
import styles from './bench.module.css';

type Hyperparameters = SubmissionMetadata['hyperparameters'];

type ParsedHyperparameters =
  | {ok: true; value: Hyperparameters}
  | {ok: false; error: string};

/** Parses the hyperparameter box: a flat JSON object of strings, numbers, and booleans. */
function parseHyperparameters(text: string): ParsedHyperparameters {
  if (text.trim() === '') {
    return {ok: true, value: {}};
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(text);
  } catch (error) {
    return {ok: false, error: `Hyperparameters are not valid JSON: ${String(error)}`};
  }
  if (typeof parsed !== 'object' || parsed === null || Array.isArray(parsed)) {
    return {ok: false, error: 'Hyperparameters must be a JSON object, for example {"k": 5}.'};
  }
  for (const [key, value] of Object.entries(parsed)) {
    if (!['string', 'number', 'boolean'].includes(typeof value)) {
      return {
        ok: false,
        error: `Hyperparameter "${key}" must be a string, number, or boolean; nested values are not accepted.`,
      };
    }
  }
  return {ok: true, value: parsed as Hyperparameters};
}

/** The name a form field has in the request, shown beside its label. */
function FieldName({children}: {children: string}): ReactNode {
  return <code className={styles.fieldName}>{children}</code>;
}

/**
 * The request the form is about to send, written as a curl command. The form and a
 * script call the same endpoint; this shows the form's current values in that shape.
 */
function EquivalentRequest({
  baseUrl,
  slug,
  metadata,
  fileName,
}: {
  baseUrl: string;
  slug: string;
  metadata: SubmissionMetadata;
  fileName: string;
}): ReactNode {
  const curl = [
    `curl -X POST ${baseUrl}/submissions \\`,
    '  -H "Authorization: Bearer $TC_BENCH_TOKEN" \\',
    `  -F dataset=${slug} \\`,
    '  -F "metadata=<metadata.json" \\',
    `  -F "predictions=@${fileName};type=text/csv"`,
  ].join('\n');
  return (
    <details className={styles.formWide}>
      <summary>This form as an API request</summary>
      <p>
        The form sends <code>POST /submissions</code> with three multipart fields. A script
        that sends the same fields with an <Link to="/benchmark/account#api-tokens">API token</Link>{' '}
        makes the same submission.
      </p>
      <CodeBlock language="json" title="metadata.json">
        {JSON.stringify(metadata, null, 2)}
      </CodeBlock>
      <CodeBlock language="bash">{curl}</CodeBlock>
    </details>
  );
}

export function QuotaPanel({quota}: {quota: Quota}): ReactNode {
  return (
    <div className={styles.panel}>
      <p className={styles.panelTitle}>Submission quota</p>
      <ul className={styles.facts}>
        <li>
          <span className={styles.factLabel}>Used</span>
          {quota.used_in_window} of {quota.max_per_window} in the last {quota.window_hours} hours
        </li>
        <li>
          <span className={styles.factLabel}>Remaining</span>
          {quota.remaining}
        </li>
        <li>
          <span className={styles.factLabel}>Minimum gap</span>
          {quota.min_gap_minutes} minutes
        </li>
        <li>
          <span className={styles.factLabel}>Next allowed</span>
          {quota.next_allowed_at ? fmtDateTime(quota.next_allowed_at) : 'now'}
        </li>
      </ul>
      <p className={clsx(styles.fieldHint)}>Rejected attempts count toward the quota.</p>
    </div>
  );
}

function ScoresTable({val, test}: {val: SplitScores; test: SplitScores}): ReactNode {
  const targets = Object.keys(val.per_target);
  const metrics = metricsIn(val);
  return (
    <>
      <div className={styles.tableWrap}>
        <table className={styles.table}>
          <thead>
            <tr>
              <th scope="col">Split (macro average)</th>
              <th scope="col" className={styles.num}>
                Records
              </th>
              {metrics.map((m) => (
                <th key={m.key} scope="col" className={styles.num}>
                  {m.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(
              [
                ['Validation', val],
                ['Test', test],
              ] as const
            ).map(([label, scores]) => (
              <tr key={label}>
                <td>{label}</td>
                <td className={styles.num}>{scores.n_records.toLocaleString('en-US')}</td>
                {metrics.map((m) => (
                  <td key={m.key} className={styles.num}>
                    {fmtMetric(scores.macro[m.key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {targets.length > 1 ? (
        <details>
          <summary>Per-target values ({targets.length} targets)</summary>
          <div className={styles.tableWrap}>
            <table className={styles.table}>
              <thead>
                <tr>
                  <th scope="col">Target</th>
                  <th scope="col">Split</th>
                  {metrics.map((m) => (
                    <th key={m.key} scope="col" className={styles.num}>
                      {m.label}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {targets.flatMap((target) =>
                  (
                    [
                      ['validation', val],
                      ['test', test],
                    ] as const
                  ).map(([label, scores]) => (
                    <tr key={`${target}-${label}`}>
                      <td>{target}</td>
                      <td>{label}</td>
                      {metrics.map((m) => (
                        <td key={m.key} className={styles.num}>
                          {fmtMetric(scores.per_target[target]?.[m.key])}
                        </td>
                      ))}
                    </tr>
                  )),
                )}
              </tbody>
            </table>
          </div>
        </details>
      ) : null}
    </>
  );
}

/** The grader's answer: scores and flags, or the list of rejection reasons. */
export function ResultPanel({result}: {result: SubmissionResult}): ReactNode {
  if (result.status === 'rejected') {
    return (
      <div className={styles.panel} role="alert">
        <p className={styles.panelTitle}>
          Submission rejected <StatusBadge status="rejected" />
        </p>
        <p>No score was recorded. The grader gave these reasons:</p>
        <ul className={styles.reasons}>
          {result.rejection_reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      </div>
    );
  }
  return (
    <div className={styles.panel} role="status">
      <p className={styles.panelTitle}>
        Submission scored <StatusBadge status={result.status} />
      </p>
      <ul className={styles.facts}>
        <li>
          <span className={styles.factLabel}>Method</span>
          {result.method_name}
        </li>
        <li>
          <span className={styles.factLabel}>Dataset</span>
          <code>{result.dataset_slug}</code>
        </li>
        <li>
          <span className={styles.factLabel}>Submission</span>
          <code>{result.submission_id}</code>
        </li>
        <li>
          <span className={styles.factLabel}>Flags</span>
          <FlagChips flags={result.flags} />
        </li>
        {result.archive_sha256 ? (
          <li>
            <span className={styles.factLabel}>Archive sha256</span>
            <code>{result.archive_sha256}</code>
          </li>
        ) : null}
      </ul>
      {result.val && result.test ? <ScoresTable val={result.val} test={result.test} /> : null}
      <p>
        <Link to={`/benchmark/leaderboard?dataset=${encodeURIComponent(result.dataset_slug)}`}>
          Open the board for this dataset
        </Link>
      </p>
    </div>
  );
}

function SubmitForm({accessToken}: {accessToken: string}): ReactNode {
  const api = useBenchApi();
  const [datasetsState, reloadDatasets] = useLoad(() => api.datasets(), [api]);
  const [quotaState, reloadQuota] = useLoad(() => api.quota(accessToken), [api, accessToken]);

  const [slug, setSlug] = useState('');
  const [methodName, setMethodName] = useState('');
  const [description, setDescription] = useState('');
  const [modelFamily, setModelFamily] = useState('');
  const [encoding, setEncoding] = useState('');
  const [codeUrl, setCodeUrl] = useState('');
  const [usesExternalData, setUsesExternalData] = useState(false);
  const [externalDataDescription, setExternalDataDescription] = useState('');
  const [hyperparameters, setHyperparameters] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [submission, setSubmission] = useState<LoadState<SubmissionResult>>({status: 'idle'});

  if (quotaState.status === 'error' && quotaState.httpStatus === 401) {
    return (
      <ErrorNotice message="Your session is no longer valid. Sign out on the account page and sign in again." />
    );
  }

  /** The metadata record as the form's fields stand now. */
  const metadataFor = (hyper: Hyperparameters): SubmissionMetadata => ({
    method_name: methodName.trim(),
    description: description.trim(),
    model_family: modelFamily.trim(),
    encoding: encoding.trim(),
    code_url: codeUrl.trim() === '' ? null : codeUrl.trim(),
    uses_external_data: usesExternalData,
    external_data_description: usesExternalData ? externalDataDescription.trim() : null,
    hyperparameters: hyper,
  });
  const parsedHyperparameters = parseHyperparameters(hyperparameters);

  const onSubmit = (event: FormEvent<HTMLFormElement>, defaultSlug: string): void => {
    event.preventDefault();
    setFormError(null);
    if (!parsedHyperparameters.ok) {
      setFormError(parsedHyperparameters.error);
      return;
    }
    if (file === null) {
      setFormError('Choose the predictions CSV file.');
      return;
    }
    const metadata = metadataFor(parsedHyperparameters.value);
    setSubmission({status: 'loading'});
    api.submit(accessToken, slug || defaultSlug, metadata, file).then(
      (result) => {
        setSubmission({status: 'ok', data: result});
        reloadQuota();
      },
      (error: unknown) => {
        setSubmission(failureState<SubmissionResult>(error));
        reloadQuota();
      },
    );
  };

  return (
    <>
      <LoadGate state={quotaState} onRetry={reloadQuota}>
        {(quota) => <QuotaPanel quota={quota} />}
      </LoadGate>
      <LoadGate state={datasetsState} onRetry={reloadDatasets}>
        {(datasets) => {
          if (datasets.length === 0) {
            return (
              <div className={styles.empty}>
                <p className={styles.emptyTitle}>No benchmark datasets are registered</p>
                <p>The API answered with an empty dataset list, so there is nothing to submit to.</p>
              </div>
            );
          }
          const blockedUntil =
            quotaState.status === 'ok' &&
            quotaState.data.next_allowed_at !== null &&
            new Date(quotaState.data.next_allowed_at).getTime() > Date.now()
              ? quotaState.data.next_allowed_at
              : null;
          const sending = submission.status === 'loading';
          return (
            <form className={styles.form} onSubmit={(e) => onSubmit(e, datasets[0].slug)}>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>
                  Dataset
                  <FieldName>dataset</FieldName>
                </span>
                <select
                  className={styles.select}
                  value={slug || datasets[0].slug}
                  onChange={(e) => setSlug(e.target.value)}>
                  {datasets.map((d) => (
                    <option key={d.slug} value={d.slug}>
                      {d.title}
                    </option>
                  ))}
                </select>
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>
                  Method name
                  <FieldName>metadata.method_name</FieldName>
                </span>
                <input
                  className={styles.input}
                  value={methodName}
                  onChange={(e) => setMethodName(e.target.value)}
                  required
                />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>
                  Model family
                  <FieldName>metadata.model_family</FieldName>
                </span>
                <input
                  className={styles.input}
                  value={modelFamily}
                  onChange={(e) => setModelFamily(e.target.value)}
                  placeholder="for example: kNN, linear, GNN, transformer"
                  required
                />
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>
                  Encoding
                  <FieldName>metadata.encoding</FieldName>
                </span>
                <input
                  className={styles.input}
                  value={encoding}
                  onChange={(e) => setEncoding(e.target.value)}
                  placeholder="the gene encoding the model reads"
                  required
                />
              </label>
              <label className={clsx(styles.field, styles.formWide)}>
                <span className={styles.fieldLabel}>
                  Description
                  <FieldName>metadata.description</FieldName>
                </span>
                <textarea
                  className={styles.textarea}
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  required
                />
              </label>
              <label className={clsx(styles.field, styles.formWide)}>
                <span className={styles.fieldLabel}>
                  Code URL (optional)
                  <FieldName>metadata.code_url</FieldName>
                </span>
                <input
                  className={styles.input}
                  type="url"
                  value={codeUrl}
                  onChange={(e) => setCodeUrl(e.target.value)}
                  placeholder="https://"
                />
                <span className={styles.fieldHint}>
                  A public repository at a fixed commit. Verification needs it.
                </span>
              </label>
              <label className={clsx(styles.checkbox, styles.formWide)}>
                <input
                  type="checkbox"
                  checked={usesExternalData}
                  onChange={(e) => setUsesExternalData(e.target.checked)}
                />
                The method uses data beyond the benchmark training split
                <FieldName>metadata.uses_external_data</FieldName>
              </label>
              {usesExternalData ? (
                <label className={clsx(styles.field, styles.formWide)}>
                  <span className={styles.fieldLabel}>
                  External data description
                  <FieldName>metadata.external_data_description</FieldName>
                </span>
                  <textarea
                    className={styles.textarea}
                    value={externalDataDescription}
                    onChange={(e) => setExternalDataDescription(e.target.value)}
                    required
                  />
                </label>
              ) : null}
              <label className={clsx(styles.field, styles.formWide)}>
                <span className={styles.fieldLabel}>
                  Hyperparameters (JSON object, optional)
                  <FieldName>metadata.hyperparameters</FieldName>
                </span>
                <textarea
                  className={clsx(styles.textarea, styles.mono)}
                  value={hyperparameters}
                  onChange={(e) => setHyperparameters(e.target.value)}
                  placeholder='{"k": 5, "metric": "cosine"}'
                  spellCheck={false}
                />
                <span className={styles.fieldHint}>
                  One level only: each value is a string, a number, or a boolean.
                </span>
              </label>
              <label className={clsx(styles.field, styles.formWide)}>
                <span className={styles.fieldLabel}>
                  Predictions CSV
                  <FieldName>predictions</FieldName>
                </span>
                <input
                  className={styles.input}
                  type="file"
                  accept=".csv,text/csv"
                  onChange={(e) => setFile(e.target.files ? e.target.files[0] ?? null : null)}
                  required
                />
                <span className={styles.fieldHint}>
                  Long format with the header <code>record_id,split,target,prediction</code>,
                  one row for every validation and test record of every target.
                </span>
              </label>
              <div className={styles.formActions}>
                <button
                  type="submit"
                  className="button button--primary"
                  disabled={sending || blockedUntil !== null}>
                  {sending ? 'Grading' : 'Submit predictions'}
                </button>
                {blockedUntil !== null ? (
                  <span className={styles.muted}>
                    Next submission allowed at {fmtDateTime(blockedUntil)}.
                  </span>
                ) : null}
              </div>
              {formError !== null ? (
                <div className={styles.formWide}>
                  <ErrorNotice message={formError} />
                </div>
              ) : null}
              <EquivalentRequest
                baseUrl={api.baseUrl}
                slug={slug || datasets[0].slug}
                metadata={metadataFor(
                  parsedHyperparameters.ok ? parsedHyperparameters.value : {},
                )}
                fileName={file ? file.name : 'predictions.csv'}
              />
            </form>
          );
        }}
      </LoadGate>
      {submission.status === 'ok' ? <ResultPanel result={submission.data} /> : null}
      {submission.status === 'error' ? (
        <ErrorNotice message={submission.message} reasons={submission.reasons} />
      ) : null}
      {submission.status === 'unreachable' ? <ApiUnreachable url={submission.url} /> : null}
    </>
  );
}

function Submit(): ReactNode {
  const session = useBenchSession();
  if (session.accessToken === null) {
    return (
      <div className={styles.empty}>
        <p className={styles.emptyTitle}>Sign in to use the form</p>
        <p>
          The form submits for the account signed in to this browser. Sign in through
          CILogon on the account page; a script uses an API token instead, as described
          above.
        </p>
        <Link className="button button--primary" to="/benchmark/account">
          Go to the account page
        </Link>
      </div>
    );
  }
  return <SubmitForm accessToken={session.accessToken} />;
}

export default function SubmitApp(): ReactNode {
  return <BenchFrame>{() => <Submit />}</BenchFrame>;
}
