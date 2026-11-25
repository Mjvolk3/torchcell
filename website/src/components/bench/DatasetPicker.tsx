import React, {
  useEffect,
  useId,
  useRef,
  useState,
  type KeyboardEvent,
  type ReactNode,
  type RefObject,
} from 'react';
import clsx from 'clsx';
import {
  BenchApiError,
  MAX_TASK_BOARDS,
  splitLabel,
  type BenchApi,
  type BenchmarkDatasetPublic,
  type TaskBoard,
  type TaskHit,
} from '@site/src/lib/benchApi';
import {failureState, useLoad} from '@site/src/lib/useLoad';
import {LoadGate} from './common';
import styles from './bench.module.css';

/** Wait after the last keystroke before searching, in milliseconds. */
const SEARCH_DEBOUNCE_MS = 150;
/** Tasks loaded per page of the task list (the first page, and each "Show more"). */
const TASK_PAGE = 20;
/** Rows PageUp and PageDown move the active option by. */
const PAGE_STEP = 5;
/** A task with more boards than this picks its split with a select, not buttons. */
const SPLIT_BUTTONS_MAX = 12;

function fmtCount(n: number): string {
  return n.toLocaleString('en-US');
}

function plural(n: number, one: string, many: string): string {
  return `${fmtCount(n)} ${n === 1 ? one : many}`;
}

/** The organism as an italic chip, or nothing when the task has none. */
function OrganismChip({organism}: {organism: string | null | undefined}): ReactNode {
  if (organism == null) {
    return null;
  }
  return <span className={clsx('tc-chip', styles.organismChip)}>{organism}</span>;
}

/** Why a search failed, in words for the list's note. */
function failureMessage(error: unknown): string {
  const failure = failureState<null>(error);
  if (failure.status === 'unreachable') {
    return `The benchmark API is not reachable at ${failure.url}`;
  }
  return failure.status === 'error' ? failure.message : 'Search failed';
}

/** The label of a board in the split control: its readable split, or its title. */
function boardLabel(board: TaskBoard): string {
  return board.split_name === null ? board.title : splitLabel(board.split_name);
}

function asTaskBoard(d: BenchmarkDatasetPublic): TaskBoard {
  return {
    slug: d.slug,
    split_name: d.split_name ?? null,
    title: d.title,
    n_train: d.n_train,
    n_val: d.n_val,
    n_test: d.n_test,
  };
}

/** The task of a board: its task group, or its own slug when it has none. */
function taskKeyOf(d: BenchmarkDatasetPublic): string {
  return d.task_group ?? d.slug;
}

/**
 * The first control: All organisms, then each organism with its task count from
 * /datasets/facets. Disabled until the facets arrive.
 */
function OrganismSelect({
  api,
  id,
  value,
  onChange,
}: {
  api: BenchApi;
  id: string;
  value: string | null;
  onChange: (organism: string | null) => void;
}): ReactNode {
  const [state, reload] = useLoad(() => api.datasetFacets(), [api]);
  if (state.status === 'unreachable' || state.status === 'error') {
    return (
      <LoadGate state={state} onRetry={reload}>
        {() => null}
      </LoadGate>
    );
  }
  const facets = state.status === 'ok' ? state.data : null;
  return (
    <select
      id={id}
      className={clsx(styles.select, styles.pickerOrganismSelect)}
      value={value ?? ''}
      disabled={facets === null}
      onChange={(e) => onChange(e.target.value === '' ? null : e.target.value)}>
      <option value="">
        All organisms{facets === null ? '' : ` (${plural(facets.n_tasks, 'task', 'tasks')})`}
      </option>
      {(facets?.organisms ?? []).map((facet) => (
        <option key={facet.organism} value={facet.organism}>
          {facet.organism} ({plural(facet.n_tasks, 'task', 'tasks')})
        </option>
      ))}
    </select>
  );
}

/** The active option and whether the keyboard put it there (only then is it scrolled to). */
type Nav = {index: number; byKey: boolean};

/**
 * The second control: an ARIA combobox over /tasks/search. Typing ranks tasks (one
 * row per task, never one per split); an empty box lists the tasks of the chosen
 * organism in task order. Either list is paged TASK_PAGE at a time with a "Show
 * more" row at its end, which loads by itself when the keyboard reaches the last
 * loaded row, so a held ArrowDown walks the whole list.
 */
function TaskCombobox({
  api,
  inputId,
  inputRef,
  organism,
  currentKey,
  currentTitle,
  onChoose,
}: {
  api: BenchApi;
  inputId: string;
  inputRef: RefObject<HTMLInputElement | null>;
  organism: string | null;
  currentKey: string | null;
  currentTitle: string;
  onChoose: (task: TaskHit) => void;
}): ReactNode {
  const baseId = useId();
  const listId = `${baseId}-list`;
  const optionId = (index: number): string => `${baseId}-option-${index}`;

  const [editing, setEditing] = useState(false);
  const [text, setText] = useState('');
  const [open, setOpen] = useState(false);
  const [nav, setNav] = useState<Nav>({index: -1, byKey: false});
  // The last answer stays listed while the next one is pending, so typing does not
  // blank the list between keystrokes.
  const [items, setItems] = useState<TaskHit[]>([]);
  const [total, setTotal] = useState(0);
  const [answered, setAnswered] = useState(false);
  const [pending, setPending] = useState(false);
  const [failed, setFailed] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  // Bumped for every new list (query, organism, open or close); an answer that
  // arrives for an older list is dropped.
  const generation = useRef(0);

  const query = text.trim();

  useEffect(() => {
    generation.current += 1;
    const mine = generation.current;
    setLoadingMore(false);
    if (!open) {
      setItems([]);
      setTotal(0);
      setAnswered(false);
      setPending(false);
      setFailed(null);
      return undefined;
    }
    setPending(true);
    const timer = window.setTimeout(() => {
      api
        .searchTasks({q: query, organism: organism ?? undefined, limit: TASK_PAGE, offset: 0})
        .then(
          (page) => {
            if (generation.current === mine) {
              setItems(page.results);
              setTotal(page.total);
              setAnswered(true);
              setPending(false);
              setFailed(null);
              setNav({index: query !== '' && page.results.length > 0 ? 0 : -1, byKey: true});
            }
          },
          (error: unknown) => {
            if (generation.current === mine) {
              setItems([]);
              setTotal(0);
              setPending(false);
              setFailed(failureMessage(error));
            }
          },
        );
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      window.clearTimeout(timer);
    };
  }, [api, query, organism, open]);

  const hasMore = answered && failed === null && items.length < total;
  const optionCount = items.length + (hasMore ? 1 : 0);

  // The next page loads when the keyboard reaches the last loaded task or the "Show
  // more" row (a click on that row counts as the keyboard).
  const wantMore =
    open && hasMore && !pending && nav.byKey && nav.index >= 0 && nav.index >= items.length - 1;
  useEffect(() => {
    if (!wantMore || loadingMore) {
      return;
    }
    const mine = generation.current;
    setLoadingMore(true);
    api
      .searchTasks({
        q: query,
        organism: organism ?? undefined,
        limit: TASK_PAGE,
        offset: items.length,
      })
      .then(
        (page) => {
          if (generation.current === mine) {
            setItems((prev) => [...prev, ...page.results]);
            setTotal(page.total);
            setLoadingMore(false);
          }
        },
        (error: unknown) => {
          if (generation.current === mine) {
            setLoadingMore(false);
            setFailed(failureMessage(error));
          }
        },
      );
    // `query`, `organism` and `api` are fixed for a generation; listing them would
    // start a second request for the same page.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wantMore, loadingMore, items.length]);

  // Keep the active option in view when the keyboard moved it; a hover never scrolls.
  useEffect(() => {
    if (open && nav.byKey && nav.index >= 0) {
      document.getElementById(optionId(nav.index))?.scrollIntoView({block: 'nearest'});
    }
    // optionId is derived from baseId, which is stable.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open, nav, optionCount]);

  const close = (): void => {
    setOpen(false);
    setNav({index: -1, byKey: false});
  };

  const finishEditing = (): void => {
    close();
    setEditing(false);
    setText('');
  };

  const choose = (task: TaskHit): void => {
    finishEditing();
    inputRef.current?.blur();
    onChoose(task);
  };

  /** Moves the active option to `to(current)`, clamped to the list: no wrap-around. */
  const move = (e: KeyboardEvent<HTMLInputElement>, to: (index: number) => number): void => {
    e.preventDefault();
    if (!open) {
      setOpen(true);
      return;
    }
    if (optionCount === 0) {
      return;
    }
    const last = optionCount - 1;
    setNav((prev) => ({index: Math.max(0, Math.min(last, to(prev.index))), byKey: true}));
  };

  const onKeyDown = (e: KeyboardEvent<HTMLInputElement>): void => {
    // Key repeat is not filtered: a held arrow keeps moving, one row per repeat.
    switch (e.key) {
      case 'ArrowDown':
        move(e, (i) => i + 1);
        break;
      case 'ArrowUp':
        move(e, (i) => i - 1);
        break;
      case 'PageDown':
        move(e, (i) => i + PAGE_STEP);
        break;
      case 'PageUp':
        move(e, (i) => i - PAGE_STEP);
        break;
      case 'Home':
        if (open && optionCount > 0) {
          move(e, () => 0);
        }
        break;
      case 'End':
        if (open && optionCount > 0) {
          move(e, () => optionCount - 1);
        }
        break;
      case 'Enter':
        if (open && nav.index >= 0 && nav.index < items.length) {
          e.preventDefault();
          choose(items[nav.index]);
        } else if (open && hasMore && nav.index === items.length) {
          // The "Show more" row: its page is already loading (wantMore).
          e.preventDefault();
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

  const hover = (index: number): void => {
    if (nav.index !== index || nav.byKey) {
      setNav({index, byKey: false});
    }
  };

  let status = '';
  if (open && answered && !pending && failed === null) {
    status =
      total === 0
        ? 'No tasks match.'
        : `${plural(total, 'task', 'tasks')}${total > items.length ? `; ${fmtCount(items.length)} listed.` : '.'}`;
  }

  return (
    <div className={styles.pickerBox}>
      <input
        ref={inputRef}
        id={inputId}
        type="text"
        className={clsx(styles.input, styles.pickerInput)}
        role="combobox"
        aria-expanded={open}
        aria-controls={listId}
        aria-autocomplete="list"
        aria-activedescendant={open && nav.index >= 0 ? optionId(nav.index) : undefined}
        autoComplete="off"
        spellCheck={false}
        placeholder={
          currentTitle !== '' ? currentTitle : 'Search tasks by name, organism, target, or citation'
        }
        value={editing ? text : currentTitle}
        onFocus={() => {
          setEditing(true);
          setText('');
          setOpen(true);
        }}
        onBlur={finishEditing}
        onChange={(e) => {
          setEditing(true);
          setText(e.target.value);
          setOpen(true);
        }}
        onKeyDown={onKeyDown}
      />
      <div className={styles.pickerPopup} hidden={!open}>
        <ul id={listId} role="listbox" aria-label="Tasks" className={styles.pickerList}>
          {items.map((hit, index) => (
            <li
              key={hit.key}
              id={optionId(index)}
              role="option"
              aria-selected={index === nav.index}
              className={clsx(
                styles.pickerOption,
                index === nav.index && styles.pickerOptionActive,
              )}
              // Keep the input focused so its blur does not close the list first.
              onMouseDown={(e) => e.preventDefault()}
              // mousemove, not mouseenter: a list scrolled under a still pointer must
              // not take the active row from the keyboard.
              onMouseMove={() => hover(index)}
              onClick={() => choose(hit)}>
              <span className={styles.optionTitle}>
                {hit.title}
                {hit.key === currentKey ? (
                  <span className={styles.optionCurrent}> (current)</span>
                ) : null}
              </span>
              <span className={styles.optionMeta}>
                <OrganismChip organism={hit.organism} />
                <span>{plural(hit.n_boards, 'split', 'splits')}</span>
              </span>
            </li>
          ))}
          {hasMore ? (
            <li
              id={optionId(items.length)}
              role="option"
              aria-selected={nav.index === items.length}
              className={clsx(
                styles.pickerOption,
                styles.pickerMore,
                nav.index === items.length && styles.pickerOptionActive,
              )}
              onMouseDown={(e) => e.preventDefault()}
              onMouseMove={() => hover(items.length)}
              onClick={() => setNav({index: items.length, byKey: true})}>
              {loadingMore
                ? 'Loading more tasks'
                : `Show more (${fmtCount(total - items.length)} more)`}
            </li>
          ) : null}
        </ul>
        {pending && items.length === 0 && failed === null ? (
          <p className={styles.pickerNote}>Searching</p>
        ) : null}
        {failed !== null ? (
          <p className={clsx(styles.pickerNote, styles.pickerNoteError)}>{failed}</p>
        ) : null}
        {!pending && failed === null && answered && total === 0 ? (
          <p className={styles.pickerNote}>No tasks match.</p>
        ) : null}
      </div>
      <p className={styles.srOnly} role="status" aria-live="polite">
        {status}
      </p>
    </div>
  );
}

/**
 * The third control: one board per split of the chosen task. Up to SPLIT_BUTTONS_MAX
 * boards are a radio group of buttons (arrow keys move and choose); more are a
 * select, with a button that pages boards past the first MAX_TASK_BOARDS.
 */
function SplitPicker({
  api,
  labelId,
  task,
  selected,
  selectedSlug,
  onSelect,
}: {
  api: BenchApi;
  labelId: string;
  task: TaskHit;
  selected: BenchmarkDatasetPublic | null;
  selectedSlug: string | null;
  onSelect: (slug: string) => void;
}): ReactNode {
  // Boards paged past the task's first MAX_TASK_BOARDS, for this task only.
  const [extra, setExtra] = useState<{key: string; boards: TaskBoard[]}>({
    key: task.key,
    boards: [],
  });
  const [moreFailed, setMoreFailed] = useState<string | null>(null);
  const [loadingMore, setLoadingMore] = useState(false);
  const buttons = useRef<(HTMLButtonElement | null)[]>([]);

  const known = [...task.boards, ...(extra.key === task.key ? extra.boards : [])];
  // A board linked by URL past the loaded pages is still listed, last.
  const boards =
    selected !== null &&
    selected.slug === selectedSlug &&
    taskKeyOf(selected) === task.key &&
    !known.some((b) => b.slug === selected.slug)
      ? [...known, asTaskBoard(selected)]
      : known;

  const loadMore = (): void => {
    if (task.task_group === null) {
      return;
    }
    setLoadingMore(true);
    api
      .searchDatasets({
        organism: task.organism ?? undefined,
        taskGroup: task.task_group,
        limit: MAX_TASK_BOARDS,
        offset: known.length,
      })
      .then(
        (page) => {
          setExtra((prev) => ({
            key: task.key,
            boards: [
              ...(prev.key === task.key ? prev.boards : []),
              ...page.results.map((hit) => asTaskBoard(hit.dataset)),
            ],
          }));
          setLoadingMore(false);
          setMoreFailed(null);
        },
        (error: unknown) => {
          setLoadingMore(false);
          setMoreFailed(failureMessage(error));
        },
      );
  };

  if (task.n_boards <= 1) {
    return null;
  }

  if (task.n_boards > SPLIT_BUTTONS_MAX) {
    const listed = boards.some((b) => b.slug === selectedSlug);
    return (
      <div className={styles.splitSelectRow}>
        <select
          aria-labelledby={labelId}
          className={clsx(styles.select, styles.splitSelect)}
          value={listed && selectedSlug !== null ? selectedSlug : ''}
          onChange={(e) => onSelect(e.target.value)}>
          {listed ? null : (
            <option value="" disabled>
              Choose a split
            </option>
          )}
          {boards.map((b) => (
            <option key={b.slug} value={b.slug}>
              {boardLabel(b)}
            </option>
          ))}
        </select>
        {known.length < task.n_boards && task.task_group !== null ? (
          <button
            type="button"
            className="button button--secondary button--sm"
            disabled={loadingMore}
            onClick={loadMore}>
            {loadingMore
              ? 'Loading splits'
              : `More splits (${fmtCount(task.n_boards - known.length)} left)`}
          </button>
        ) : null}
        {moreFailed !== null ? (
          <span className={clsx(styles.pickerNote, styles.pickerNoteError)}>{moreFailed}</span>
        ) : null}
      </div>
    );
  }

  const checkedIndex = boards.findIndex((b) => b.slug === selectedSlug);
  const pick = (index: number): void => {
    buttons.current[index]?.focus();
    if (boards[index].slug !== selectedSlug) {
      onSelect(boards[index].slug);
    }
  };
  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>, index: number): void => {
    const last = boards.length - 1;
    const to: Record<string, number> = {
      ArrowRight: Math.min(index + 1, last),
      ArrowDown: Math.min(index + 1, last),
      ArrowLeft: Math.max(index - 1, 0),
      ArrowUp: Math.max(index - 1, 0),
      Home: 0,
      End: last,
    };
    if (e.key in to) {
      e.preventDefault();
      pick(to[e.key]);
    }
  };

  return (
    <div role="radiogroup" aria-labelledby={labelId} className={styles.splitGroup}>
      {boards.map((b, index) => {
        const checked = index === checkedIndex;
        return (
          <button
            key={b.slug}
            ref={(el) => {
              buttons.current[index] = el;
            }}
            type="button"
            role="radio"
            aria-checked={checked}
            // Roving focus: Tab enters the group at the chosen split (or the first).
            tabIndex={checked || (checkedIndex < 0 && index === 0) ? 0 : -1}
            className={clsx(styles.splitButton, checked && styles.splitButtonChecked)}
            title={b.split_name ?? b.title}
            onClick={() => pick(index)}
            onKeyDown={(e) => onKeyDown(e, index)}>
            {boardLabel(b)}
          </button>
        );
      })}
    </div>
  );
}

/**
 * Picks the board of the leaderboard in three steps that read left to right:
 * organism, task, split. The organism filters the task search; choosing a task
 * opens its first split; the split row lists the task's boards.
 *
 * The task shown is resolved from the selected board with /tasks/search?key=, and
 * kept while another split of it loads, so switching splits does not reload it.
 */
export default function DatasetPicker({
  api,
  selected,
  selectedSlug,
  onSelect,
}: {
  api: BenchApi;
  /** The last board that loaded, kept while the next one loads; null before any. */
  selected: BenchmarkDatasetPublic | null;
  /** The board asked for (the URL's), which may still be loading. */
  selectedSlug: string | null;
  onSelect: (slug: string) => void;
}): ReactNode {
  const baseId = useId();
  const organismId = `${baseId}-organism`;
  const taskId = `${baseId}-task`;
  const splitLabelId = `${baseId}-split`;
  const inputRef = useRef<HTMLInputElement>(null);

  const [task, setTask] = useState<TaskHit | null>(null);
  const [organism, setOrganism] = useState<string | null>(null);
  const [organismFailed, setOrganismFailed] = useState<string | null>(null);

  const show = (next: TaskHit): void => {
    setTask(next);
    setOrganism(next.organism);
  };

  // The task of the selected board is fetched only when the board is not one the
  // shown task already holds, and only once the board asked for has loaded (a stale
  // board would fetch the task being left).
  const holdsSelected =
    task !== null &&
    (task.boards.some((b) => b.slug === selectedSlug) ||
      (selected !== null && selected.slug === selectedSlug && taskKeyOf(selected) === task.key));
  const resolveKey =
    !holdsSelected && selected !== null && selected.slug === selectedSlug
      ? taskKeyOf(selected)
      : null;
  const [taskState, reloadTask] = useLoad(
    resolveKey === null
      ? null
      : () =>
          api.searchTasks({key: resolveKey, limit: 1}).then((page) => {
            const hit = page.results[0];
            if (hit === undefined) {
              throw new BenchApiError(404, `No task has the key ${resolveKey}`, [], null);
            }
            return hit;
          }),
    [api, resolveKey],
  );
  useEffect(() => {
    if (taskState.status === 'ok') {
      show(taskState.data);
    }
    // `show` only sets state; the effect runs once per answer.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [taskState]);

  const chooseTask = (next: TaskHit): void => {
    show(next);
    onSelect(next.boards[0].slug);
  };

  // A named organism other than the shown task's opens that organism's first task,
  // so the three controls always describe the board below. "All organisms" only
  // widens the task search.
  const chooseOrganism = (next: string | null): void => {
    setOrganism(next);
    setOrganismFailed(null);
    if (next === null || next === task?.organism) {
      return;
    }
    api.searchTasks({organism: next, limit: 1}).then(
      (page) => {
        const first = page.results[0];
        if (first !== undefined) {
          chooseTask(first);
        }
      },
      (error: unknown) => setOrganismFailed(failureMessage(error)),
    );
  };

  const currentTitle = task?.title ?? selected?.title ?? '';
  const board = selected !== null && selected.slug === selectedSlug ? selected : null;

  return (
    <div className={styles.picker}>
      <div className={styles.pickerRow}>
        <div className={clsx(styles.field, styles.pickerOrganism)}>
          <label htmlFor={organismId} className={styles.fieldLabel}>
            Organism
          </label>
          <OrganismSelect api={api} id={organismId} value={organism} onChange={chooseOrganism} />
        </div>
        <div className={clsx(styles.field, styles.pickerTask)}>
          <label htmlFor={taskId} className={styles.fieldLabel}>
            Task
          </label>
          <TaskCombobox
            api={api}
            inputId={taskId}
            inputRef={inputRef}
            organism={organism}
            currentKey={task?.key ?? null}
            currentTitle={currentTitle}
            onChoose={chooseTask}
          />
        </div>
        {task !== null && task.n_boards > 1 ? (
          <div className={clsx(styles.field, styles.pickerSplit)}>
            <span id={splitLabelId} className={styles.fieldLabel}>
              Split ({fmtCount(task.n_boards)})
            </span>
            <SplitPicker
              api={api}
              labelId={splitLabelId}
              task={task}
              selected={selected}
              selectedSlug={selectedSlug}
              onSelect={onSelect}
            />
          </div>
        ) : null}
      </div>
      {taskState.status === 'unreachable' || taskState.status === 'error' ? (
        <LoadGate state={taskState} onRetry={reloadTask}>
          {() => null}
        </LoadGate>
      ) : null}
      {organismFailed !== null ? (
        <p className={clsx(styles.pickerNote, styles.pickerNoteError)}>{organismFailed}</p>
      ) : null}
      {board !== null ? (
        <p className={styles.pickerSelected}>
          <code>{board.slug}</code>
          <span>
            train {fmtCount(board.n_train)} / val {fmtCount(board.n_val)} / test{' '}
            {fmtCount(board.n_test)}
          </span>
        </p>
      ) : null}
    </div>
  );
}
