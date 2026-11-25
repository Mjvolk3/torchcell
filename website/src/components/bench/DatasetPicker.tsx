import React, {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
} from 'react';
import clsx from 'clsx';
import type {
  BenchApi,
  BenchmarkDatasetPublic,
  OrganismFacet,
  SearchPage,
  TaskGroupFacet,
} from '@site/src/lib/benchApi';
import {failureState, useLoad, type LoadState} from '@site/src/lib/useLoad';
import {LoadGate} from './common';
import styles from './bench.module.css';

/** Wait after the last keystroke before searching, in milliseconds. */
const SEARCH_DEBOUNCE_MS = 150;
/** Ranked results shown under the search box. */
const SEARCH_RESULTS = 20;
/** Boards loaded per "Show more" in the browse tree. */
const BROWSE_PAGE = 50;
/** Task groups (or organisms) rendered per "Show more" in the browse tree. */
const GROUP_PAGE = 50;

function fmtCount(n: number): string {
  return n.toLocaleString('en-US');
}

/** Train, validation and test record counts in one short line. */
function splitCounts(d: BenchmarkDatasetPublic): string {
  return `train ${fmtCount(d.n_train)} / val ${fmtCount(d.n_val)} / test ${fmtCount(d.n_test)}`;
}

/** The organism as an italic chip, or nothing when the dataset has none. */
function OrganismChip({organism}: {organism: string | null | undefined}): ReactNode {
  if (organism == null) {
    return null;
  }
  return <span className={clsx('tc-chip', styles.organismChip)}>{organism}</span>;
}

/**
 * A <details> whose content is mounted only while it is open, so a closed branch of
 * the browse tree sends no request and renders no rows.
 */
function Disclosure({
  summary,
  children,
  className,
}: {
  summary: ReactNode;
  children: () => ReactNode;
  className?: string;
}): ReactNode {
  const [open, setOpen] = useState(false);
  return (
    <details
      className={className}
      onToggle={(e) => setOpen(e.currentTarget.open)}>
      <summary className={styles.browseSummary}>{summary}</summary>
      {open ? children() : null}
    </details>
  );
}

/** A choosable board in the browse tree: title, split, counts. */
function BoardButton({
  dataset,
  current,
  onSelect,
}: {
  dataset: BenchmarkDatasetPublic;
  current: boolean;
  onSelect: (dataset: BenchmarkDatasetPublic) => void;
}): ReactNode {
  return (
    <li>
      <button
        type="button"
        className={clsx(styles.boardButton, current && styles.boardButtonCurrent)}
        aria-current={current ? 'true' : undefined}
        onClick={() => onSelect(dataset)}>
        <span className={styles.optionTitle}>{dataset.title}</span>
        <span className={styles.optionMeta}>
          {dataset.split_name ? <code>{dataset.split_name}</code> : null}
          <span>{splitCounts(dataset)}</span>
        </span>
      </button>
    </li>
  );
}

/**
 * The boards of one task group, fetched from /datasets/search with the organism and
 * task-group filters, BROWSE_PAGE at a time.
 */
function TaskGroupBoards({
  api,
  organism,
  group,
  selectedSlug,
  onSelect,
}: {
  api: BenchApi;
  organism: string;
  group: TaskGroupFacet;
  selectedSlug: string | null;
  onSelect: (dataset: BenchmarkDatasetPublic) => void;
}): ReactNode {
  const [boards, setBoards] = useState<BenchmarkDatasetPublic[]>([]);
  const [total, setTotal] = useState<number | null>(null);
  const [state, setState] = useState<LoadState<null>>({status: 'loading'});
  const [offset, setOffset] = useState(0);
  // Bumped by Retry, so a failed page is requested again at the same offset.
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let cancelled = false;
    setState({status: 'loading'});
    api
      .searchDatasets({organism, taskGroup: group.task_group, limit: BROWSE_PAGE, offset})
      .then(
        (page) => {
          if (!cancelled) {
            setBoards((prev) => [...prev, ...page.results.map((hit) => hit.dataset)]);
            setTotal(page.total);
            setState({status: 'ok', data: null});
          }
        },
        (error: unknown) => {
          if (!cancelled) {
            setState(failureState<null>(error));
          }
        },
      );
    return () => {
      cancelled = true;
    };
  }, [api, organism, group.task_group, offset, attempt]);

  return (
    <>
      {boards.length > 0 ? (
        <ul className={styles.boardList}>
          {boards.map((d) => (
            <BoardButton
              key={d.slug}
              dataset={d}
              current={d.slug === selectedSlug}
              onSelect={onSelect}
            />
          ))}
        </ul>
      ) : null}
      {state.status === 'ok' ? null : (
        <LoadGate state={state} onRetry={() => setAttempt((n) => n + 1)}>
          {() => null}
        </LoadGate>
      )}
      {state.status === 'ok' && total !== null && boards.length < total ? (
        <button
          type="button"
          className="button button--secondary button--sm"
          onClick={() => setOffset(boards.length)}>
          Show more ({fmtCount(total - boards.length)} left)
        </button>
      ) : null}
    </>
  );
}

/** One organism: its task groups, GROUP_PAGE at a time. */
function OrganismBranch({
  api,
  facet,
  selectedSlug,
  onSelect,
}: {
  api: BenchApi;
  facet: OrganismFacet;
  selectedSlug: string | null;
  onSelect: (dataset: BenchmarkDatasetPublic) => void;
}): ReactNode {
  const [shown, setShown] = useState(GROUP_PAGE);
  const groups = facet.task_groups.slice(0, shown);
  return (
    <>
      <ul className={styles.browseList}>
        {groups.map((group) => (
          <li key={group.task_group}>
            <Disclosure
              summary={
                <>
                  {group.title} <span className={styles.browseCount}>{fmtCount(group.count)}</span>
                </>
              }>
              {() => (
                <TaskGroupBoards
                  api={api}
                  organism={facet.organism}
                  group={group}
                  selectedSlug={selectedSlug}
                  onSelect={onSelect}
                />
              )}
            </Disclosure>
          </li>
        ))}
      </ul>
      {facet.task_groups.length > shown ? (
        <button
          type="button"
          className="button button--secondary button--sm"
          onClick={() => setShown((n) => n + GROUP_PAGE)}>
          Show more tasks ({fmtCount(facet.task_groups.length - shown)} left)
        </button>
      ) : null}
      {facet.n_without_task_group > 0 ? (
        <p className={styles.browseNote}>
          {fmtCount(facet.n_without_task_group)}{' '}
          {facet.n_without_task_group === 1 ? 'board has' : 'boards have'} no task group;
          find {facet.n_without_task_group === 1 ? 'it' : 'them'} with the search box.
        </p>
      ) : null}
    </>
  );
}

/** Organisms, then task groups, then boards, each level opened on demand. */
function BrowseTree({
  api,
  selectedSlug,
  onSelect,
}: {
  api: BenchApi;
  selectedSlug: string | null;
  onSelect: (dataset: BenchmarkDatasetPublic) => void;
}): ReactNode {
  const [state, reload] = useLoad(() => api.datasetFacets(), [api]);
  const [shown, setShown] = useState(GROUP_PAGE);
  return (
    <LoadGate state={state} onRetry={reload}>
      {(facets) => (
        <div className={styles.browseTree}>
          {facets.organisms.length === 0 ? (
            <p className={styles.browseNote}>No dataset names its organism yet.</p>
          ) : (
            <ul className={styles.browseList}>
              {facets.organisms.slice(0, shown).map((facet) => (
                <li key={facet.organism}>
                  <Disclosure
                    summary={
                      <>
                        <em>{facet.organism}</em>{' '}
                        <span className={styles.browseCount}>{fmtCount(facet.count)}</span>
                      </>
                    }>
                    {() => (
                      <OrganismBranch
                        api={api}
                        facet={facet}
                        selectedSlug={selectedSlug}
                        onSelect={onSelect}
                      />
                    )}
                  </Disclosure>
                </li>
              ))}
            </ul>
          )}
          {facets.organisms.length > shown ? (
            <button
              type="button"
              className="button button--secondary button--sm"
              onClick={() => setShown((n) => n + GROUP_PAGE)}>
              Show more organisms ({fmtCount(facets.organisms.length - shown)} left)
            </button>
          ) : null}
          {facets.n_without_organism > 0 ? (
            <p className={styles.browseNote}>
              {fmtCount(facets.n_without_organism)}{' '}
              {facets.n_without_organism === 1 ? 'board has' : 'boards have'} no organism
              set; find {facets.n_without_organism === 1 ? 'it' : 'them'} with the search box.
            </p>
          ) : null}
        </div>
      )}
    </LoadGate>
  );
}

/** Why a search failed, in words for the list's note. */
function failureMessage(error: unknown): string {
  const failure = failureState<SearchPage>(error);
  if (failure.status === 'unreachable') {
    return `The benchmark API is not reachable at ${failure.url}`;
  }
  return failure.status === 'error' ? failure.message : 'Search failed';
}

/**
 * Picks the dataset of the leaderboard. A search box (an ARIA combobox) asks
 * /datasets/search as the user types and lists up to SEARCH_RESULTS ranked boards;
 * the Browse section walks organism, task group and board, loading each level only
 * when it is opened. Neither renders more than a page of rows at once.
 */
export default function DatasetPicker({
  api,
  selected,
  onSelect,
}: {
  api: BenchApi;
  /** The dataset on the board now, or null while none is loaded. */
  selected: BenchmarkDatasetPublic | null;
  onSelect: (dataset: BenchmarkDatasetPublic) => void;
}): ReactNode {
  const baseId = useId();
  const inputId = `${baseId}-input`;
  const listId = `${baseId}-list`;
  const optionId = (index: number): string => `${baseId}-option-${index}`;
  const inputRef = useRef<HTMLInputElement>(null);

  const [editing, setEditing] = useState(false);
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(-1);
  // The last answer stays listed while the next one is pending, so typing does not
  // blank the list between keystrokes.
  const [page, setPage] = useState<SearchPage | null>(null);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);

  const query = text.trim();

  useEffect(() => {
    if (!open || query === '') {
      setPage(null);
      setPending(false);
      setFailed(null);
      return undefined;
    }
    let cancelled = false;
    setPending(true);
    const timer = window.setTimeout(() => {
      api.searchDatasets({q: query, limit: SEARCH_RESULTS}).then(
        (answer) => {
          if (!cancelled) {
            setPage(answer);
            setPending(false);
            setFailed(null);
            setActive(answer.results.length > 0 ? 0 : -1);
          }
        },
        (error: unknown) => {
          if (!cancelled) {
            setPending(false);
            setFailed(failureMessage(error));
          }
        },
      );
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [api, query, open]);

  const results = failed === null && page !== null ? page.results : [];

  const close = (): void => {
    setOpen(false);
    setActive(-1);
  };

  const finishEditing = (): void => {
    close();
    setEditing(false);
    setText('');
  };

  const choose = (dataset: BenchmarkDatasetPublic): void => {
    finishEditing();
    inputRef.current?.blur();
    onSelect(dataset);
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    switch (e.key) {
      case 'ArrowDown':
        e.preventDefault();
        if (!open) {
          setOpen(true);
        } else if (results.length > 0) {
          setActive((i) => Math.min(i + 1, results.length - 1));
        }
        break;
      case 'ArrowUp':
        e.preventDefault();
        if (results.length > 0) {
          setActive((i) => Math.max(i - 1, 0));
        }
        break;
      case 'Enter':
        if (open && active >= 0 && active < results.length) {
          e.preventDefault();
          choose(results[active].dataset);
        }
        break;
      case 'Escape':
        e.preventDefault();
        if (open) {
          close();
        } else {
          // Leaving the box ends the edit (onBlur) and shows the current title again.
          inputRef.current?.blur();
        }
        break;
      default:
        break;
    }
  };

  const showList = open && query !== '';
  let status = '';
  if (showList && page !== null && !pending && failed === null) {
    const {total} = page;
    status =
      total === 0
        ? 'No datasets match.'
        : `${fmtCount(total)} ${total === 1 ? 'dataset matches' : 'datasets match'}` +
          (total > results.length ? `; showing the best ${results.length}.` : '.');
  }

  return (
    <div className={styles.picker}>
      <label htmlFor={inputId} className={styles.fieldLabel}>
        Dataset
      </label>
      <div className={styles.pickerBox}>
        <input
          ref={inputRef}
          id={inputId}
          type="text"
          className={clsx(styles.input, styles.pickerInput)}
          role="combobox"
          aria-expanded={showList}
          aria-controls={listId}
          aria-autocomplete="list"
          aria-activedescendant={showList && active >= 0 ? optionId(active) : undefined}
          autoComplete="off"
          spellCheck={false}
          placeholder={
            selected ? selected.title : 'Search by name, organism, task, split, or citation'
          }
          value={editing ? text : (selected?.title ?? '')}
          onFocus={() => {
            setEditing(true);
            setText('');
          }}
          onBlur={finishEditing}
          onChange={(e) => {
            setEditing(true);
            setText(e.target.value);
            setOpen(true);
            setActive(-1);
          }}
          onKeyDown={onKeyDown}
        />
        <div className={styles.pickerPopup} hidden={!showList}>
          <ul id={listId} role="listbox" aria-label="Matching datasets" className={styles.pickerList}>
            {results.map((hit, index) => (
              <li
                key={hit.dataset.slug}
                id={optionId(index)}
                role="option"
                aria-selected={index === active}
                className={clsx(
                  styles.pickerOption,
                  index === active && styles.pickerOptionActive,
                )}
                // Keep the input focused so its blur does not close the list first.
                onMouseDown={(e) => e.preventDefault()}
                onMouseEnter={() => setActive(index)}
                onClick={() => choose(hit.dataset)}>
                <span className={styles.optionTitle}>{hit.dataset.title}</span>
                <span className={styles.optionMeta}>
                  <OrganismChip organism={hit.dataset.organism} />
                  {hit.dataset.split_name ? <code>{hit.dataset.split_name}</code> : null}
                  <span>{splitCounts(hit.dataset)}</span>
                </span>
              </li>
            ))}
          </ul>
          {pending && results.length === 0 ? (
            <p className={styles.pickerNote}>Searching</p>
          ) : null}
          {failed !== null ? (
            <p className={clsx(styles.pickerNote, styles.pickerNoteError)}>{failed}</p>
          ) : null}
          {!pending && failed === null && page !== null && page.total === 0 ? (
            <p className={styles.pickerNote}>No datasets match.</p>
          ) : null}
        </div>
      </div>
      <p className={styles.srOnly} role="status" aria-live="polite">
        {status}
      </p>
      {selected ? (
        <p className={styles.pickerSelected}>
          <OrganismChip organism={selected.organism} />
          {selected.split_name ? <code>{selected.split_name}</code> : null}
          <code>{selected.slug}</code>
        </p>
      ) : null}
      <Disclosure className={styles.browse} summary="Browse by organism and task">
        {() => (
          <BrowseTree api={api} selectedSlug={selected?.slug ?? null} onSelect={onSelect} />
        )}
      </Disclosure>
    </div>
  );
}
