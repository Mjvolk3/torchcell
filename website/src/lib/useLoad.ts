import {useCallback, useEffect, useState, type DependencyList} from 'react';
import {BenchApiError, BenchApiUnreachableError} from './benchApi';

export type LoadState<T> =
  | {status: 'idle'}
  | {status: 'loading'}
  | {status: 'ok'; data: T}
  | {status: 'unreachable'; url: string}
  | {status: 'error'; message: string; reasons: string[]; httpStatus: number | null};

/** Maps a thrown value to the matching non-ok LoadState. */
export function failureState<T>(error: unknown): LoadState<T> {
  if (error instanceof BenchApiUnreachableError) {
    return {status: 'unreachable', url: error.url};
  }
  if (error instanceof BenchApiError) {
    // A rate-limit answer (HTTP 429) carries the time of the next allowed attempt.
    const reasons =
      error.nextAllowedAt === null
        ? error.reasons
        : [
            ...error.reasons,
            `Next submission allowed at ${new Date(error.nextAllowedAt).toLocaleString()}`,
          ];
    return {status: 'error', message: error.message, reasons, httpStatus: error.status};
  }
  return {status: 'error', message: String(error), reasons: [], httpStatus: null};
}

/**
 * Runs `load` when `deps` change and tracks its state. Passing `null` as the loader
 * leaves the state idle (for example while no dataset is selected). `reload` runs the
 * current loader again.
 */
export function useLoad<T>(
  load: (() => Promise<T>) | null,
  deps: DependencyList,
): [LoadState<T>, () => void] {
  const [state, setState] = useState<LoadState<T>>({status: load ? 'loading' : 'idle'});
  const [nonce, setNonce] = useState(0);

  useEffect(() => {
    if (load === null) {
      setState({status: 'idle'});
      return undefined;
    }
    let cancelled = false;
    setState({status: 'loading'});
    load().then(
      (data) => {
        if (!cancelled) {
          setState({status: 'ok', data});
        }
      },
      (error: unknown) => {
        if (!cancelled) {
          setState(failureState<T>(error));
        }
      },
    );
    return () => {
      cancelled = true;
    };
    // The loader closes over `deps`; listing it would rerun on every render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);

  const reload = useCallback(() => setNonce((n) => n + 1), []);
  return [state, reload];
}
