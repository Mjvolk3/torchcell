import React, {type ReactNode} from 'react';
import Link from '@docusaurus/Link';
import {metricInfo, type BenchmarkDatasetPublic} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useLoad} from '@site/src/lib/useLoad';
import {BenchFrame, LoadGate} from './common';
import styles from './bench.module.css';

/** The pages under docs/benchmark/datasets/, by slug; a dataset without one links to the board. */
const DATASET_PAGES: Record<string, string> = {
  'gene-essentiality-sgd': '/benchmark/datasets/gene-essentiality-sgd',
};

/** A task with several boards (one per split): every slug that starts with the task name shares its page. */
const TASK_PAGES: Record<string, string> = {
  'ecoli-growth-rate-campos2018': '/benchmark/datasets/ecoli/growth-rate-campos2018',
  'ecoli-morphology-campos2018': '/benchmark/datasets/ecoli/morphology-campos2018',
  'ecoli-gene-interaction-babu2014': '/benchmark/datasets/ecoli/gene-interaction-babu2014',
  'ecoli-metabolome-fuhrer2017': '/benchmark/datasets/ecoli/metabolome-fuhrer2017',
};

function pageOf(slug: string): string | undefined {
  if (DATASET_PAGES[slug]) {
    return DATASET_PAGES[slug];
  }
  const task = Object.keys(TASK_PAGES).find((name) => slug.startsWith(`${name}-`));
  return task ? TASK_PAGES[task] : undefined;
}

function Table({datasets}: {datasets: BenchmarkDatasetPublic[]}): ReactNode {
  const sorted = [...datasets].sort((a, b) => a.task.localeCompare(b.task) || a.slug.localeCompare(b.slug));
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Dataset</th>
            <th scope="col">Task</th>
            <th scope="col">Version</th>
            <th scope="col" className={styles.num}>
              Train
            </th>
            <th scope="col" className={styles.num}>
              Val
            </th>
            <th scope="col" className={styles.num}>
              Test
            </th>
            <th scope="col">Primary metric</th>
            <th scope="col">Sources</th>
            <th scope="col">Board</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((d) => {
            const page = pageOf(d.slug);
            return (
              <tr key={d.slug}>
                <td>
                  {page ? <Link to={page}>{d.title}</Link> : d.title}
                  <br />
                  <code>{d.slug}</code>
                </td>
                <td>
                  {d.task}
                  {d.targets.length > 1 ? `, ${d.targets.length} targets` : ''}
                </td>
                <td>{d.version}</td>
                <td className={styles.num}>{d.n_train.toLocaleString('en-US')}</td>
                <td className={styles.num}>{d.n_val.toLocaleString('en-US')}</td>
                <td className={styles.num}>{d.n_test.toLocaleString('en-US')}</td>
                <td>{metricInfo(d.primary_metric).label}</td>
                <td>
                  {d.provenance
                    ? d.provenance.sources.map((s) => s.role).join('; ')
                    : <span className={styles.muted}>not recorded</span>}
                </td>
                <td>
                  <Link to={`/benchmark/leaderboard?dataset=${encodeURIComponent(d.slug)}`}>
                    leaderboard
                  </Link>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

function Datasets(): ReactNode {
  const api = useBenchApi();
  const [state, reload] = useLoad(() => api.datasets(), [api]);
  return (
    <LoadGate state={state} onRetry={reload}>
      {(datasets) =>
        datasets.length === 0 ? (
          <p className={styles.muted}>No dataset is served yet.</p>
        ) : (
          <Table datasets={datasets} />
        )
      }
    </LoadGate>
  );
}

/** Every served benchmark dataset, read from the API at page load, one row each. */
export default function DatasetsTable(): ReactNode {
  return <BenchFrame>{() => <Datasets />}</BenchFrame>;
}
