import React, {useMemo, useState, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import type {Data, Layout} from 'plotly.js';
import {
  MACRO_TARGET,
  metricInfo,
  metricsFor,
  metricValue,
  type BenchApi,
  type BenchmarkDatasetPublic,
  type LeaderboardRow,
  type MetricName,
} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useLoad} from '@site/src/lib/useLoad';
import StatusBadge from '@site/src/components/StatusBadge';
import Plot, {baseLayout, SERIES_COLORS, usePlotTheme, type PlotTheme} from './Plot';
import {BenchFrame, CodeLink, FlagChips, LoadGate, fmtDate, fmtMetric, isHttpUrl} from './common';
import styles from './bench.module.css';

type RankedRow = LeaderboardRow & {
  rank: number;
  valValue: number | null;
  testValue: number | null;
};

type SortKey =
  | 'rank'
  | 'method'
  | 'user'
  | 'encoding'
  | 'family'
  | 'status'
  | 'val'
  | 'test'
  | 'flags'
  | 'date';

type SortState = {key: SortKey; ascending: boolean};

/**
 * Ranks rows by the selected test metric, best first (highest for Pearson, Spearman,
 * and R2; lowest for MSE and MAE). Rows with no value for the selection rank last.
 */
export function rankRows(rows: LeaderboardRow[], metric: MetricName, target: string): RankedRow[] {
  const {higherIsBetter} = metricInfo(metric);
  const withValues = rows.map((row) => ({
    ...row,
    valValue: metricValue(row.val, metric, target),
    testValue: metricValue(row.test, metric, target),
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
    case 'test':
      return compareNullable(a.testValue, b.testValue);
    case 'flags':
      return a.flags.length - b.flags.length;
    case 'date':
      return a.submitted_at.localeCompare(b.submitted_at);
  }
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
    {name: 'Submission, provisional', baseline: false, status: 'provisional', symbol: 'circle'},
    {name: 'Submission, verified', baseline: false, status: 'verified', symbol: 'diamond'},
    {name: 'Baseline, provisional', baseline: true, status: 'provisional', symbol: 'circle-open'},
    {name: 'Baseline, verified', baseline: true, status: 'verified', symbol: 'diamond-open'},
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
}: {
  label: string;
  sortKey: SortKey;
  sort: SortState;
  onSort: (key: SortKey) => void;
  numeric?: boolean;
}): ReactNode {
  const active = sort.key === sortKey;
  return (
    <th
      scope="col"
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

export function BoardTable({rows, metricLabel}: {rows: RankedRow[]; metricLabel: string}): ReactNode {
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

  const header = (label: string, key: SortKey, numeric?: boolean): ReactNode => (
    <SortHeader label={label} sortKey={key} sort={sort} onSort={onSort} numeric={numeric} />
  );

  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            {header('Rank', 'rank', true)}
            {header('Method', 'method')}
            {header('User', 'user')}
            {header('Encoding', 'encoding')}
            {header('Model family', 'family')}
            {header('Status', 'status')}
            {header(`Val ${metricLabel}`, 'val', true)}
            {header(`Test ${metricLabel}`, 'test', true)}
            {header('Flags', 'flags')}
            {header('Date', 'date')}
            <th scope="col">Code</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.submission_id} className={clsx(row.is_baseline && styles.baselineRow)}>
              <td className={styles.num}>{row.rank}</td>
              <td>
                {row.method_name} {row.is_baseline ? <StatusBadge status="baseline" /> : null}
              </td>
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
              <td className={styles.num}>{fmtMetric(row.valValue)}</td>
              <td className={styles.num}>{fmtMetric(row.testValue)}</td>
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

export function DatasetPanel({dataset, api}: {dataset: BenchmarkDatasetPublic; api: BenchApi}): ReactNode {
  const splitsUrl = api.splitsUrl(dataset.slug);
  const templateUrl = api.templateUrl(dataset.slug);
  return (
    <div className={styles.panel}>
      <p className={styles.panelTitle}>{dataset.title}</p>
      <p>{dataset.description}</p>
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
        const ranked = rankRows(rows, metric, target);
        return (
          <>
            <BoardTable rows={ranked} metricLabel={metricLabel} />
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

function Leaderboard(): ReactNode {
  const api = useBenchApi();
  const location = useLocation();
  const requestedSlug = new URLSearchParams(location.search).get('dataset');

  const [datasetsState, reloadDatasets] = useLoad(() => api.datasets(), [api]);
  const [slug, setSlug] = useState<string | null>(requestedSlug);
  // Null means the dataset's primary metric; a choice lasts until the dataset changes.
  const [chosenMetric, setMetric] = useState<MetricName | null>(null);
  const [target, setTarget] = useState<string>(MACRO_TARGET);
  const [verifiedOnly, setVerifiedOnly] = useState(false);

  return (
    <LoadGate state={datasetsState} onRetry={reloadDatasets}>
      {(datasets) => {
        if (datasets.length === 0) {
          return (
            <div className={styles.empty}>
              <p className={styles.emptyTitle}>No benchmark datasets are registered</p>
              <p>The API answered with an empty dataset list.</p>
            </div>
          );
        }
        const dataset = datasets.find((d) => d.slug === slug) ?? datasets[0];
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
                <span className={styles.fieldLabel}>Dataset</span>
                <select
                  className={styles.select}
                  value={dataset.slug}
                  onChange={(e) => {
                    setSlug(e.target.value);
                    setTarget(MACRO_TARGET);
                    setMetric(null);
                  }}>
                  {datasets.map((d) => (
                    <option key={d.slug} value={d.slug}>
                      {d.title}
                    </option>
                  ))}
                </select>
              </label>
              <label className={styles.field}>
                <span className={styles.fieldLabel}>Metric</span>
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
  );
}

export default function LeaderboardApp(): ReactNode {
  return <BenchFrame>{() => <Leaderboard />}</BenchFrame>;
}
