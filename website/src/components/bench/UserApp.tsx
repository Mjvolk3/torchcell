import React, {useState, type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import type {Data, Layout} from 'plotly.js';
import {
  METRICS,
  metricInfo,
  metricValue,
  type MetricName,
  type UserHistory,
  type UserHistoryRow,
} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useLoad} from '@site/src/lib/useLoad';
import StatusBadge from '@site/src/components/StatusBadge';
import Plot, {baseLayout, SERIES_COLORS, usePlotTheme, type PlotTheme} from './Plot';
import {BenchFrame, CodeLink, FlagChips, LoadGate, fmtDate, fmtMetric} from './common';
import styles from './bench.module.css';

function byDataset(rows: UserHistoryRow[]): [string, UserHistoryRow[]][] {
  const groups = new Map<string, UserHistoryRow[]>();
  for (const row of rows) {
    const group = groups.get(row.dataset_slug);
    if (group) {
      group.push(row);
    } else {
      groups.set(row.dataset_slug, [row]);
    }
  }
  return [...groups.entries()].sort(([a], [b]) => a.localeCompare(b));
}

/**
 * Validation and test over time for one dataset. The dataset sets the color;
 * validation is a solid line with circles, test a dashed line with diamonds.
 */
function DatasetHistoryChart({
  slug,
  rows,
  metric,
  color,
  theme,
}: {
  slug: string;
  rows: UserHistoryRow[];
  metric: MetricName;
  color: string;
  theme: PlotTheme;
}): ReactNode {
  const metricLabel = metricInfo(metric).label;
  const ordered = [...rows].sort((a, b) => a.submitted_at.localeCompare(b.submitted_at));
  const x = ordered.map((r) => r.submitted_at);
  const customdata = ordered.map((r) => [r.method_name, fmtDate(r.submitted_at), r.status]);
  const hover = (split: string): string =>
    `<b>%{customdata[0]}</b><br>%{customdata[1]}, %{customdata[2]}<br>${split} %{y:.4f}<extra></extra>`;

  const data: Data[] = [
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Validation',
      x,
      y: ordered.map((r) => metricValue(r.val, metric)),
      customdata,
      line: {color, width: 2},
      marker: {symbol: 'circle', size: 9, color, line: {color: theme.surface, width: 1.5}},
      hovertemplate: hover('validation'),
    },
    {
      type: 'scatter',
      mode: 'lines+markers',
      name: 'Test',
      x,
      y: ordered.map((r) => metricValue(r.test, metric)),
      customdata,
      line: {color, width: 2, dash: 'dash'},
      marker: {symbol: 'diamond-open', size: 10, color, line: {color, width: 2}},
      hovertemplate: hover('test'),
    },
  ];

  const base = baseLayout(theme);
  const layout: Partial<Layout> = {
    ...base,
    xaxis: {...base.xaxis, type: 'date', title: {text: 'Submission date'}},
    yaxis: {...base.yaxis, title: {text: metricLabel}},
  };

  return (
    <div className={styles.chart}>
      <p className={styles.chartTitle}>
        <code>{slug}</code>
      </p>
      <p className={styles.chartNote}>
        Validation (solid, circles) and test (dashed, open diamonds) {metricLabel} for each
        scored submission, in time order.
      </p>
      <Plot
        data={data}
        layout={layout}
        ariaLabel={`Validation and test ${metricLabel} over time on ${slug}`}
      />
    </div>
  );
}

function HistoryTable({rows, metric}: {rows: UserHistoryRow[]; metric: MetricName}): ReactNode {
  const metricLabel = metricInfo(metric).label;
  const sorted = [...rows].sort((a, b) => b.submitted_at.localeCompare(a.submitted_at));
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Dataset</th>
            <th scope="col">Method</th>
            <th scope="col">Encoding</th>
            <th scope="col">Model family</th>
            <th scope="col">Status</th>
            <th scope="col" className={styles.num}>
              Val {metricLabel}
            </th>
            <th scope="col" className={styles.num}>
              Test {metricLabel}
            </th>
            <th scope="col">Flags</th>
            <th scope="col">Code</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.submission_id}>
              <td title={row.submitted_at}>{fmtDate(row.submitted_at)}</td>
              <td>
                <Link to={`/benchmark/leaderboard?dataset=${encodeURIComponent(row.dataset_slug)}`}>
                  {row.dataset_slug}
                </Link>
              </td>
              <td>
                {row.method_name} {row.is_baseline ? <StatusBadge status="baseline" /> : null}
              </td>
              <td>{row.encoding}</td>
              <td>{row.model_family}</td>
              <td>
                <StatusBadge status={row.status} />
              </td>
              <td className={styles.num}>{fmtMetric(metricValue(row.val, metric))}</td>
              <td className={styles.num}>{fmtMetric(metricValue(row.test, metric))}</td>
              <td>
                <FlagChips flags={row.flags} />
              </td>
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

export function History({history}: {history: UserHistory}): ReactNode {
  const theme = usePlotTheme();
  const [metric, setMetric] = useState<MetricName>('pearson');
  const {user, submissions} = history;
  const groups = byDataset(submissions);

  return (
    <>
      <div className={styles.panel}>
        <p className={styles.panelTitle}>{user.display_name}</p>
        <ul className={styles.facts}>
          <li>
            <span className={styles.factLabel}>Affiliation</span>
            {user.affiliation ?? 'none'}
          </li>
          <li>
            <span className={styles.factLabel}>Signs in through</span>
            {user.identity_provider ?? 'not applicable'}
          </li>
          <li>
            <span className={styles.factLabel}>Joined</span>
            {fmtDate(user.created_at)}
          </li>
          <li>
            <span className={styles.factLabel}>Scored submissions</span>
            {submissions.length}
          </li>
        </ul>
      </div>
      {submissions.length === 0 ? (
        <div className={styles.empty}>
          <p className={styles.emptyTitle}>No scored submissions</p>
          <p>This user has no scored submissions.</p>
        </div>
      ) : (
        <>
          <div className={styles.controls}>
            <label className={styles.field}>
              <span className={styles.fieldLabel}>Metric (macro average over targets)</span>
              <select
                className={styles.select}
                value={metric}
                onChange={(e) => setMetric(e.target.value as MetricName)}>
                {METRICS.map((m) => (
                  <option key={m.key} value={m.key}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
          </div>
          <HistoryTable rows={submissions} metric={metric} />
          <div className={styles.charts}>
            {groups.map(([slug, rows], index) => (
              <DatasetHistoryChart
                key={slug}
                slug={slug}
                rows={rows}
                metric={metric}
                color={SERIES_COLORS[Math.min(index, SERIES_COLORS.length - 1)]}
                theme={theme}
              />
            ))}
          </div>
        </>
      )}
    </>
  );
}

function User(): ReactNode {
  const api = useBenchApi();
  const location = useLocation();
  const userId = new URLSearchParams(location.search).get('id');
  const [state, reload] = useLoad(
    userId === null ? null : () => api.userHistory(userId),
    [api, userId],
  );

  if (userId === null) {
    return (
      <div className={styles.empty}>
        <p className={styles.emptyTitle}>No user selected</p>
        <p>
          This page takes a user id in the address, as <code>?id=&lt;user_id&gt;</code>. User
          names on the <Link to="/benchmark/leaderboard">leaderboard</Link> link here.
        </p>
      </div>
    );
  }
  return (
    <LoadGate state={state} onRetry={reload}>
      {(history) => <History history={history} />}
    </LoadGate>
  );
}

export default function UserApp(): ReactNode {
  return <BenchFrame>{() => <User />}</BenchFrame>;
}
