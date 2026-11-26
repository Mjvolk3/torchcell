import React, {useEffect, useMemo, useState, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useHistory, useLocation} from '@docusaurus/router';
import type {Data, Layout} from 'plotly.js';
import {
  MACRO_TARGET,
  metricInfo,
  metricsFor,
  metricValue,
  type BenchApi,
  type BenchmarkDatasetPublic,
  type BundleProvenance,
  DATA_SCOPE_LABELS,
  DATA_SCOPE_TITLES,
  type LeaderboardRow,
  type MetricInfo,
  type MetricName,
} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useLoad} from '@site/src/lib/useLoad';
import StatusBadge from '@site/src/components/StatusBadge';
import DatasetPicker from './DatasetPicker';
import Plot, {baseLayout, SERIES_COLORS, usePlotTheme, type PlotTheme} from './Plot';
import {BenchFrame, CodeLink, FlagChips, LoadGate, fmtDate, fmtMetric, isHttpUrl} from './common';
import styles from './bench.module.css';

type RankedRow = LeaderboardRow & {
  rank: number;
  valValue: number | null;
  testValue: number | null;
  /** The test value of every metric of the task, for the selected target. */
  testValues: Partial<Record<MetricName, number | null>>;
};

type SortKey =
  | 'rank'
  | 'method'
  | 'user'
  | 'encoding'
  | 'family'
  | 'status'
  | 'val'
  | 'flags'
  | 'date'
  | `test:${MetricName}`;

type SortState = {key: SortKey; ascending: boolean};

/**
 * Ranks rows by the selected test metric, best first (highest for Pearson, Spearman,
 * and R2; lowest for RMSE, MSE and MAE). Rows with no value for the selection rank
 * last. Every row also carries its test value of each metric in `metrics`.
 */
export function rankRows(
  rows: LeaderboardRow[],
  metric: MetricName,
  target: string,
  metrics: readonly MetricInfo[] = [metricInfo(metric)],
): RankedRow[] {
  const {higherIsBetter} = metricInfo(metric);
  const withValues = rows.map((row) => ({
    ...row,
    valValue: metricValue(row.val, metric, target),
    testValue: metricValue(row.test, metric, target),
    testValues: Object.fromEntries(
      metrics.map((m) => [m.key, metricValue(row.test, m.key, target)]),
    ) as Partial<Record<MetricName, number | null>>,
  }));
  withValues.sort((a, b) => {
    if (a.testValue === null || b.testValue === null) {
      return Number(a.testValue === null) - Number(b.testValue === null);
    }
    return higherIsBetter ? b.testValue - a.testValue : a.testValue - b.testValue;
  });
  return withValues.map((row, index) => ({...row, rank: index + 1}));
}

function compareNullable(a: number | null, b: number | null): number {
  if (a === null || b === null) {
    return Number(a === null) - Number(b === null);
  }
  return a - b;
}

function compareRows(a: RankedRow, b: RankedRow, key: SortKey): number {
  switch (key) {
    case 'rank':
      return a.rank - b.rank;
    case 'method':
      return a.method_name.localeCompare(b.method_name);
    case 'user':
      return a.display_name.localeCompare(b.display_name);
    case 'encoding':
      return a.encoding.localeCompare(b.encoding);
    case 'family':
      return a.model_family.localeCompare(b.model_family);
    case 'status':
      return a.status.localeCompare(b.status);
    case 'val':
      return compareNullable(a.valValue, b.valValue);
    case 'flags':
      return a.flags.length - b.flags.length;
    case 'date':
      return a.submitted_at.localeCompare(b.submitted_at);
    default: {
      const metric = key.slice('test:'.length) as MetricName;
      return compareNullable(a.testValues[metric] ?? null, b.testValues[metric] ?? null);
    }
  }
}

/** The best test value of each metric over the rows; null when no row has one. */
export function bestTestValues(
  rows: RankedRow[],
  metrics: readonly MetricInfo[],
): Partial<Record<MetricName, number | null>> {
  return Object.fromEntries(
    metrics.map((m) => {
      const values = rows
        .map((row) => row.testValues[m.key] ?? null)
        .filter((v): v is number => v !== null);
      if (values.length === 0) {
        return [m.key, null];
      }
      return [m.key, m.higherIsBetter ? Math.max(...values) : Math.min(...values)];
    }),
  );
}

function userHref(userId: string): string {
  return `/benchmark/user?id=${encodeURIComponent(userId)}`;
}

// ---------------------------------------------------------------------------
// Charts
// ---------------------------------------------------------------------------

type PlottedRow = RankedRow & {valValue: number; testValue: number};

function hoverData(rows: RankedRow[]): string[][] {
  return rows.map((r) => [r.method_name, r.display_name, fmtDate(r.submitted_at)]);
}

/**
 * Validation against test, one point per submission, with the y = x line. Kind and
 * status each use two channels: submissions are filled orange and baselines open
 * gray; provisional is a circle and verified a diamond.
 */
export function ValTestScatter({
  rows,
  metricLabel,
  theme,
}: {
  rows: RankedRow[];
  metricLabel: string;
  theme: PlotTheme;
}): ReactNode {
  const plotted = rows.filter(
    (r): r is PlottedRow => r.valValue !== null && r.testValue !== null,
  );
  if (plotted.length === 0) {
    return <p className={styles.muted}>No submissions to plot.</p>;
  }

  const values = plotted.flatMap((r) => [r.valValue, r.testValue]);
  const lo = Math.min(...values);
  const hi = Math.max(...values);
  const pad = (hi - lo || Math.abs(hi) || 1) * 0.08;
  const range: [number, number] = [lo - pad, hi + pad];

  const groups = [
    // Fill says status (open = provisional, filled = verified); shape says who
    // (circle = submission, diamond = baseline).
    {name: 'Submission, provisional', baseline: false, status: 'provisional', symbol: 'circle-open'},
    {name: 'Submission, verified', baseline: false, status: 'verified', symbol: 'circle'},
    {name: 'Baseline, provisional', baseline: true, status: 'provisional', symbol: 'diamond-open'},
    {name: 'Baseline, verified', baseline: true, status: 'verified', symbol: 'diamond'},
  ] as const;

  const data: Data[] = [
    {
      type: 'scatter',
      mode: 'lines',
      name: 'y = x',
      x: range,
      y: range,
      line: {color: theme.muted, width: 1, dash: 'dot'},
      hoverinfo: 'skip',
    },
  ];
  for (const group of groups) {
    const members = plotted.filter(
      (r) => r.is_baseline === group.baseline && r.status === group.status,
    );
    if (members.length === 0) {
      continue;
    }
    data.push({
      type: 'scatter',
      mode: 'markers',
      name: group.name,
      x: members.map((r) => r.valValue),
      y: members.map((r) => r.testValue),
      customdata: hoverData(members),
      marker: {
        symbol: group.symbol,
        size: 11,
        color: group.baseline ? theme.neutral : SERIES_COLORS[0],
        line: {
          color: group.baseline ? theme.neutral : theme.surface,
          width: group.baseline ? 2 : 1.5,
        },
      },
      hovertemplate:
        '<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}' +
        '<br>validation %{x:.4f}<br>test %{y:.4f}<extra>%{fullData.name}</extra>',
    });
  }

  const base = baseLayout(theme);
  const layout: Partial<Layout> = {
    ...base,
    xaxis: {...base.xaxis, range, title: {text: `Validation ${metricLabel}`}},
    yaxis: {...base.yaxis, range, title: {text: `Test ${metricLabel}`}},
  };
  return (
    <Plot
      data={data}
      layout={layout}
      ariaLabel={`Scatter of validation against test ${metricLabel}, one point per submission`}
    />
  );
}

/**
 * Running best test value over submission time, as a step line. Baselines are left
 * out of the running best and drawn as one dashed reference level.
 */
export function BestOverTime({
  rows,
  metric,
  metricLabel,
  theme,
}: {
  rows: RankedRow[];
  metric: MetricName;
  metricLabel: string;
  theme: PlotTheme;
}): ReactNode {
  const {higherIsBetter} = metricInfo(metric);
  const better = (a: number, b: number): boolean => (higherIsBetter ? a > b : a < b);
  const plotted = rows.filter((r): r is PlottedRow => r.valValue !== null && r.testValue !== null);
  const submissions = plotted
    .filter((r) => !r.is_baseline)
    .sort((a, b) => a.submitted_at.localeCompare(b.submitted_at));
  if (submissions.length === 0) {
    return <p className={styles.muted}>No non-baseline submissions to plot.</p>;
  }

  const stepX: string[] = [];
  const stepY: number[] = [];
  const improvements: PlottedRow[] = [];
  let best: number | null = null;
  for (const row of submissions) {
    if (best === null || better(row.testValue, best)) {
      best = row.testValue;
      improvements.push(row);
    }
    stepX.push(row.submitted_at);
    stepY.push(best);
  }

  const data: Data[] = [
    {
      type: 'scatter',
      mode: 'lines',
      name: 'Best submission',
      x: stepX,
      y: stepY,
      line: {color: SERIES_COLORS[0], width: 2, shape: 'hv'},
      hoverinfo: 'skip',
    },
    {
      type: 'scatter',
      mode: 'markers',
      name: 'New best',
      showlegend: false,
      x: improvements.map((r) => r.submitted_at),
      y: improvements.map((r) => r.testValue),
      customdata: hoverData(improvements),
      marker: {
        symbol: improvements.map((r) => (r.status === 'verified' ? 'diamond' : 'circle')),
        size: 10,
        color: SERIES_COLORS[0],
        line: {color: theme.surface, width: 1.5},
      },
      hovertemplate:
        '<b>%{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}' +
        '<br>test %{y:.4f}<extra></extra>',
    },
  ];

  const baselines = plotted.filter((r) => r.is_baseline);
  if (baselines.length > 0) {
    const bestBaseline = baselines.reduce((acc, r) =>
      better(r.testValue, acc.testValue) ? r : acc,
    );
    const dates = plotted.map((r) => r.submitted_at).sort();
    data.push({
      type: 'scatter',
      mode: 'lines',
      name: `Best baseline (${bestBaseline.method_name})`,
      x: [dates[0], dates[dates.length - 1]],
      y: [bestBaseline.testValue, bestBaseline.testValue],
      line: {color: theme.neutral, width: 2, dash: 'dash'},
      hovertemplate: `${bestBaseline.method_name}<br>test %{y:.4f}<extra></extra>`,
    });
  }

  const base = baseLayout(theme);
  const layout: Partial<Layout> = {
    ...base,
    xaxis: {...base.xaxis, type: 'date', title: {text: 'Submission date'}},
    yaxis: {...base.yaxis, title: {text: `Best test ${metricLabel}`}},
  };
  return (
    <Plot
      data={data}
      layout={layout}
      ariaLabel={`Step line of the best test ${metricLabel} over submission date`}
    />
  );
}

// ---------------------------------------------------------------------------
// Table
// ---------------------------------------------------------------------------

function SortHeader({
  label,
  sortKey,
  sort,
  onSort,
  numeric,
  rowSpan,
  title,
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (key: SortKey) => void;
  numeric?: boolean;
  rowSpan?: number;
  title?: string;
}): ReactNode {
  const active = sort.key === sortKey;
  return (
    <th
      scope="col"
      rowSpan={rowSpan}
      title={title}
      className={clsx(numeric && styles.num)}
      aria-sort={active ? (sort.ascending ? 'ascending' : 'descending') : 'none'}>
      <button type="button" className={styles.sortButton} onClick={() => onSort(sortKey)}>
        {label}
        <span className={styles.sortArrow} aria-hidden="true">
          {active ? (sort.ascending ? '▲' : '▼') : '▴▾'}
        </span>
      </button>
    </th>
  );
}

/**
 * The board: one row per submission, with the test value of every metric of the task
 * side by side and the validation value of the ranking metric. The best test value of
 * each metric is bold.
 */
export function BoardTable({
  rows,
  metric,
  metrics,
}: {
  rows: RankedRow[];
  metric: MetricName;
  metrics: readonly MetricInfo[];
}): ReactNode {
  const metricLabel = metricInfo(metric).label;
  const best = useMemo(() => bestTestValues(rows, metrics), [rows, metrics]);
  const [sort, setSort] = useState<SortState>({key: 'rank', ascending: true});
  const onSort = (key: SortKey): void =>
    setSort((prev) => ({key, ascending: prev.key === key ? !prev.ascending : true}));

  const sorted = useMemo(() => {
    const copy = [...rows];
    copy.sort((a, b) => {
      const order = compareRows(a, b, sort.key);
      return sort.ascending ? order : -order;
    });
    return copy;
  }, [rows, sort]);

  // Every column outside the test group spans both header rows.
  const header = (label: string, key: SortKey, numeric?: boolean): ReactNode => (
    <SortHeader
      label={label}
      sortKey={key}
      sort={sort}
      onSort={onSort}
      numeric={numeric}
      rowSpan={2}
    />
  );

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {header('Rank', 'rank', true)}
            {header('Method', 'method')}
            {header(`Val ${metricLabel}`, 'val', true)}
            <th scope="colgroup" colSpan={metrics.length} className={styles.groupHead}>
              Test
            </th>
            {header('User', 'user')}
            {header('Encoding', 'encoding')}
            {header('Model family', 'family')}
            {header('Status', 'status')}
            <th scope="col" rowSpan={2}>
              Data
            </th>
            {header('Flags', 'flags')}
            {header('Date', 'date')}
            <th scope="col" rowSpan={2}>
              Code
            </th>
          </tr>
          <tr>
            {metrics.map((m) => (
              <SortHeader
                key={m.key}
                label={m.label}
                sortKey={`test:${m.key}`}
                sort={sort}
                onSort={onSort}
                numeric
                title={`${m.higherIsBetter ? 'Higher' : 'Lower'} is better${
                  m.key === metric ? '; the board is ranked by this metric' : ''
                }`}
              />
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.submission_id} className={clsx(row.is_baseline && styles.baselineRow)}>
              <td className={styles.num}>{row.rank}</td>
              <td>
                {row.method_name} {row.is_baseline ? <StatusBadge status="baseline" /> : null}
              </td>
              <td className={styles.num}>{fmtMetric(row.valValue)}</td>
              {metrics.map((m) => {
                const value = row.testValues[m.key] ?? null;
                return (
                  <td
                    key={m.key}
                    className={clsx(
                      styles.num,
                      m.key === metric && styles.rankMetric,
                      value !== null && value === best[m.key] && styles.bestValue,
                    )}>
                    {fmtMetric(value)}
                  </td>
                );
              })}
              <td>
                <Link to={userHref(row.user_id)} title={row.affiliation ?? undefined}>
                  {row.display_name}
                </Link>
              </td>
              <td>{row.encoding}</td>
              <td>{row.model_family}</td>
              <td>
                <StatusBadge status={row.status} />
              </td>
              <td>
                {row.data_scope ? (
                  <span
                    className={clsx('tc-chip', `tc-chip--scope-${row.data_scope}`)}
                    title={DATA_SCOPE_TITLES[row.data_scope]}
                  >
                    {DATA_SCOPE_LABELS[row.data_scope]}
                  </span>
                ) : (
                  <span className={styles.muted} title="Submitted before the data scope field existed">
                    undeclared
                  </span>
                )}
              </td>
              <td>
                <FlagChips flags={row.flags} />
              </td>
              <td title={row.submitted_at}>{fmtDate(row.submitted_at)}</td>
              <td>
                <CodeLink url={row.code_url} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

const GITHUB_TREE = 'https://github.com/Mjvolk3/torchcell/blob/main/';

/**
 * Where the bundle's records and labels came from, first in the panel: every source
 * with its retrieval and hash, then the label and split rules, then the script.
 */
function Provenance({provenance}: {provenance: BundleProvenance}): ReactNode {
  return (
    <details className={styles.provenance}>
      <summary className={styles.panelTitle}>
        Provenance: {provenance.sources.length} sources, label rule, split rule
      </summary>
      <ul className={styles.facts}>
        {provenance.sources.map((s) => (
          <li key={s.name}>
            <span className={styles.factLabel}>{s.role}</span>
            {s.name}
            {s.n_files !== null ? `, ${s.n_files.toLocaleString('en-US')} files` : ''}
            {s.retrieved_at ? `, retrieved ${s.retrieved_at.slice(0, 10)}` : ''}
            {' '}({s.retrieval_method}
            {s.source_url && isHttpUrl(s.source_url) ? (
              <>
                {', '}
                <a href={s.source_url} target="_blank" rel="noopener noreferrer">
                  source
                </a>
              </>
            ) : null}
            ), sha256 <code title={s.sha256}>{s.sha256.slice(0, 12)}</code>
            {s.note ? <span className={styles.muted}> {s.note}</span> : null}
          </li>
        ))}
        <li>
          <span className={styles.factLabel}>Label rule</span>
          {provenance.label_rule}
        </li>
        <li>
          <span className={styles.factLabel}>Split rule</span>
          {provenance.split_rule}
        </li>
        <li>
          <span className={styles.factLabel}>Built by</span>
          <a href={GITHUB_TREE + provenance.script} target="_blank" rel="noopener noreferrer">
            <code>{provenance.script}</code>
          </a>
        </li>
        {provenance.notes.map((note) => (
          <li key={note} className={styles.muted}>
            {note}
          </li>
        ))}
      </ul>
    </details>
  );
}

export function DatasetPanel({dataset, api}: {dataset: BenchmarkDatasetPublic; api: BenchApi}): ReactNode {
  const splitsUrl = api.splitsUrl(dataset.slug);
  const templateUrl = api.templateUrl(dataset.slug);
  return (
    <div className={styles.panel}>
      <p className={styles.panelTitle}>{dataset.title}</p>
      <p>{dataset.description}</p>
      {dataset.provenance ? <Provenance provenance={dataset.provenance} /> : null}
      <ul className={styles.facts}>
        <li>
          <span className={styles.factLabel}>Loader</span>
          <code>{dataset.loader_class}</code>
        </li>
        <li>
          <span className={styles.factLabel}>Version</span>
          {dataset.version}
        </li>
        <li>
          <span className={styles.factLabel}>Records</span>
          train {dataset.n_train.toLocaleString('en-US')}, validation{' '}
          {dataset.n_val.toLocaleString('en-US')}, test {dataset.n_test.toLocaleString('en-US')}
        </li>
        <li>
          <span className={styles.factLabel}>Targets</span>
          {dataset.targets.length}
        </li>
        <li>
          <span className={styles.factLabel}>Task</span>
          {dataset.task === 'binary'
            ? 'binary: predict a score, larger meaning more likely 1'
            : 'regression: predict the value'}
        </li>
        <li>
          <span className={styles.factLabel}>Primary metric</span>
          {metricInfo(dataset.primary_metric).label}
        </li>
      </ul>
      <ul className={styles.facts}>
        {splitsUrl ? (
          <li>
            <a href={splitsUrl}>Download the split file (splits.csv)</a>
          </li>
        ) : (
          <li className={styles.muted}>Split file: not available in mock mode</li>
        )}
        {templateUrl ? (
          <li>
            <a href={templateUrl}>Download the predictions template (template.csv)</a>
          </li>
        ) : (
          <li className={styles.muted}>Predictions template: not available in mock mode</li>
        )}
        {dataset.docs_url && isHttpUrl(dataset.docs_url) ? (
          <li>
            <a href={dataset.docs_url} target="_blank" rel="noopener noreferrer">
              Dataset page
            </a>
          </li>
        ) : null}
      </ul>
    </div>
  );
}

function Board({
  api,
  dataset,
  metric,
  target,
  verifiedOnly,
}: {
  api: BenchApi;
  dataset: BenchmarkDatasetPublic;
  metric: MetricName;
  target: string;
  verifiedOnly: boolean;
}): ReactNode {
  const theme = usePlotTheme();
  const [state, reload] = useLoad(
    () => api.leaderboard(dataset.slug, verifiedOnly),
    [api, dataset.slug, verifiedOnly],
  );
  const metricLabel = metricInfo(metric).label;

  return (
    <LoadGate state={state} onRetry={reload}>
      {(rows) => {
        if (rows.length === 0) {
          return (
            <div className={styles.empty}>
              <p className={styles.emptyTitle}>No submissions on this board yet</p>
              <p>
                {verifiedOnly
                  ? 'No verified submissions exist for this dataset. Turn off the verified-only filter to see provisional ones.'
                  : 'The API returned no rows for this dataset.'}
              </p>
            </div>
          );
        }
        const metrics = metricsFor(dataset.task);
        const ranked = rankRows(rows, metric, target, metrics);
        return (
          <>
            <p className={styles.chartNote}>
              Ranked by test {metricLabel}; the Rank by selector changes it
              and the charts. Every test metric is shown, with the best value of each in
              bold. Select a column heading to sort by it.
            </p>
            <BoardTable rows={ranked} metric={metric} metrics={metrics} />
            <div className={styles.charts}>
              <div className={styles.chart}>
                <p className={styles.chartTitle}>Validation against test</p>
                <p className={styles.chartNote}>
                  One point per submission. Points above the dotted line score higher on test
                  than on validation.
                </p>
                <ValTestScatter rows={ranked} metricLabel={metricLabel} theme={theme} />
              </div>
              <div className={styles.chart}>
                <p className={styles.chartTitle}>Best test {metricLabel} over time</p>
                <p className={styles.chartNote}>
                  Running best among non-baseline submissions; a marker shows each new best.
                </p>
                <BestOverTime
                  rows={ranked}
                  metric={metric}
                  metricLabel={metricLabel}
                  theme={theme}
                />
              </div>
            </div>
          </>
        );
      }}
    </LoadGate>
  );
}

/** The `?dataset=<slug>` of the page URL, or null when absent. */
function slugFromSearch(search: string): string | null {
  return new URLSearchParams(search).get('dataset');
}

/**
 * The dataset the URL names, or, with no `?dataset=`, the first board (in split
 * order) of the first task in the API's task order (organism, title); null when no
 * dataset is registered.
 */
function loadSelected(api: BenchApi, slug: string | null): Promise<BenchmarkDatasetPublic | null> {
  if (slug !== null) {
    return api.dataset(slug);
  }
  return api.searchTasks({limit: 1}).then((page) => {
    const first = page.results[0];
    return first === undefined ? null : api.dataset(first.boards[0].slug);
  });
}

function Leaderboard(): ReactNode {
  const api = useBenchApi();
  const location = useLocation();
  const history = useHistory();
  // The URL holds the selection, so a board can be linked and the back button works.
  const slug = slugFromSearch(location.search);

  const [datasetState, reloadDataset] = useLoad(() => loadSelected(api, slug), [api, slug]);
  // Null means the dataset's primary metric; a choice lasts until the dataset changes.
  const [chosenMetric, setMetric] = useState<MetricName | null>(null);
  const [target, setTarget] = useState<string>(MACRO_TARGET);
  const [verifiedOnly, setVerifiedOnly] = useState(false);

  const choose = (nextSlug: string): void => {
    setTarget(MACRO_TARGET);
    setMetric(null);
    if (nextSlug !== slug) {
      const params = new URLSearchParams(location.search);
      params.set('dataset', nextSlug);
      history.push({pathname: location.pathname, search: `?${params.toString()}`});
    }
  };

  // The last dataset that loaded, kept while the next one loads, so the picker keeps
  // its organism, task and split instead of blanking on every split change.
  const [shown, setShown] = useState<BenchmarkDatasetPublic | null>(null);
  useEffect(() => {
    if (datasetState.status === 'ok') {
      setShown(datasetState.data);
    }
  }, [datasetState]);

  return (
    <>
      <div className={styles.controls}>
        <DatasetPicker
          api={api}
          selected={shown}
          selectedSlug={slug ?? shown?.slug ?? null}
          onSelect={choose}
        />
      </div>
      <LoadGate state={datasetState} onRetry={reloadDataset}>
        {(dataset) => {
          if (dataset === null) {
            return (
              <div className={styles.empty}>
                <p className={styles.emptyTitle}>No benchmark datasets are registered</p>
                <p>The API answered with an empty dataset list.</p>
              </div>
            );
          }
          const activeTarget = dataset.targets.includes(target) ? target : MACRO_TARGET;
          const metrics = metricsFor(dataset.task);
          const metric =
            chosenMetric !== null && metrics.some((m) => m.key === chosenMetric)
              ? chosenMetric
              : dataset.primary_metric;
          return (
            <>
              <div className={styles.controls}>
                <label className={styles.field}>
                  <span className={styles.fieldLabel}>Rank by</span>
                  <select
                    className={styles.select}
                    value={metric}
                    onChange={(e) => setMetric(e.target.value as MetricName)}>
                    {metrics.map((m) => (
                      <option key={m.key} value={m.key}>
                        {m.label}
                      </option>
                    ))}
                  </select>
                </label>
                {dataset.targets.length > 1 ? (
                  <label className={styles.field}>
                    <span className={styles.fieldLabel}>Target</span>
                    <select
                      className={styles.select}
                      value={activeTarget}
                      onChange={(e) => setTarget(e.target.value)}>
                      <option value={MACRO_TARGET}>Macro average over targets</option>
                      {dataset.targets.map((t) => (
                        <option key={t} value={t}>
                          {t}
                        </option>
                      ))}
                    </select>
                  </label>
                ) : null}
                <label className={styles.checkbox}>
                  <input
                    type="checkbox"
                    checked={verifiedOnly}
                    onChange={(e) => setVerifiedOnly(e.target.checked)}
                  />
                  Verified only
                </label>
              </div>
              <DatasetPanel dataset={dataset} api={api} />
              <Board
                api={api}
                dataset={dataset}
                metric={metric}
                target={activeTarget}
                verifiedOnly={verifiedOnly}
              />
            </>
          );
        }}
      </LoadGate>
    </>
  );
}

export default function LeaderboardApp(): ReactNode {
  return <BenchFrame>{() => <Leaderboard />}</BenchFrame>;
}
