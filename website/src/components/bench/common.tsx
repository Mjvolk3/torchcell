import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import BrowserOnly from '@docusaurus/BrowserOnly';
import {useBenchConfig} from '@site/src/lib/useBenchApi';
import type {LoadState} from '@site/src/lib/useLoad';
import styles from './bench.module.css';

/** Four decimals, or "n/a" when the value is missing or not finite. */
export function fmtMetric(value: number | null | undefined): string {
  return typeof value === 'number' && Number.isFinite(value) ? value.toFixed(4) : 'n/a';
}

/** The calendar date of an ISO timestamp, as written by the API (YYYY-MM-DD). */
export function fmtDate(iso: string): string {
  return iso.slice(0, 10);
}

/** Local date and time for an ISO timestamp. */
export function fmtDateTime(iso: string): string {
  return new Date(iso).toLocaleString();
}

/**
 * Frame for an interactive benchmark page. Everything inside runs only in the
 * browser. In mock mode it puts the mock notice above the content, in addition to the
 * fixed bar that Root renders.
 */
export function BenchFrame({children}: {children: () => ReactNode}): ReactNode {
  const {benchApiMock} = useBenchConfig();
  return (
    <>
      {benchApiMock ? (
        <div className={clsx(styles.notice, styles.noticeMock)} role="status">
          MOCK DATA, not real results. This build reads the fixtures in static/mock/ and
          sends nothing to the benchmark API.
        </div>
      ) : null}
      <BrowserOnly fallback={<p className={styles.muted}>Loading</p>}>{children}</BrowserOnly>
    </>
  );
}

/** Empty state for an API that did not answer. Shows no rows, real or invented. */
export function ApiUnreachable({url, onRetry}: {url: string; onRetry?: () => void}): ReactNode {
  return (
    <div className={styles.empty} role="alert">
      <p className={styles.emptyTitle}>The benchmark API is not reachable at {url}</p>
      <p>
        No results are shown because none could be loaded. The API may be down, or the
        site may have been built with a different <code>BENCH_API_URL</code>.
      </p>
      {onRetry ? (
        <button type="button" className="button button--secondary button--sm" onClick={onRetry}>
          Retry
        </button>
      ) : null}
    </div>
  );
}

export function ErrorNotice({message, reasons}: {message: string; reasons?: string[]}): ReactNode {
  return (
    <div className={clsx(styles.notice, styles.noticeError)} role="alert">
      {message}
      {reasons && reasons.length > 0 ? (
        <ul className={styles.reasons}>
          {reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
    </div>
  );
}

type GateProps<T> = {
  state: LoadState<T>;
  onRetry?: () => void;
  children: (data: T) => ReactNode;
};

/** Renders `children` once a load succeeded, and the matching state otherwise. */
export function LoadGate<T>({state, onRetry, children}: GateProps<T>): ReactNode {
  switch (state.status) {
    case 'idle':
      return null;
    case 'loading':
      return <p className={styles.muted}>Loading</p>;
    case 'unreachable':
      return <ApiUnreachable url={state.url} onRetry={onRetry} />;
    case 'error':
      return <ErrorNotice message={state.message} reasons={state.reasons} />;
    case 'ok':
      return children(state.data);
  }
}

export function FlagChips({flags}: {flags: string[]}): ReactNode {
  if (flags.length === 0) {
    return <span className={styles.muted}>none</span>;
  }
  return (
    <span className={styles.chips}>
      {flags.map((flag) => (
        <span key={flag} className="tc-chip tc-chip--flag">
          {flag}
        </span>
      ))}
    </span>
  );
}

/** True for http and https URLs only; submitter-supplied links are rendered only then. */
export function isHttpUrl(url: string): boolean {
  return /^https?:\/\//i.test(url);
}

export function CodeLink({url}: {url: string | null}): ReactNode {
  if (url === null) {
    return <span className={styles.muted}>none</span>;
  }
  if (!isHttpUrl(url)) {
    return <span className={styles.muted}>{url}</span>;
  }
  return (
    <a href={url} target="_blank" rel="noopener noreferrer">
      code
    </a>
  );
}
