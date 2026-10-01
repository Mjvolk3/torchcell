import React, {useEffect, useState, type FormEvent, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import {
  type BenchApi,
  type MessageResponse,
  type SubmissionResult,
  type TokenResponse,
} from '@site/src/lib/benchApi';
import {useBenchApi} from '@site/src/lib/useBenchApi';
import {useBenchSession, type BenchSession} from '@site/src/lib/benchAuth';
import {failureState, useLoad, type LoadState} from '@site/src/lib/useLoad';
import StatusBadge from '@site/src/components/StatusBadge';
import {
  ApiUnreachable,
  BenchFrame,
  ErrorNotice,
  FlagChips,
  LoadGate,
  fmtDate,
  fmtDateTime,
  fmtMetric,
} from './common';
import styles from './bench.module.css';

/** Shows the outcome of a form request: nothing while idle, then the message or error. */
function RequestOutcome({state}: {state: LoadState<MessageResponse>}): ReactNode {
  switch (state.status) {
    case 'idle':
    case 'loading':
      return null;
    case 'ok':
      return (
        <div className={clsx(styles.notice, styles.noticeOk)} role="status">
          {state.data.message}
        </div>
      );
    case 'unreachable':
      return <ApiUnreachable url={state.url} />;
    case 'error':
      return <ErrorNotice message={state.message} reasons={state.reasons} />;
  }
}

/** Confirms an email address when the page is opened with ?verify=<token>. */
function VerifyEmail({api, token}: {api: BenchApi; token: string}): ReactNode {
  const [state] = useLoad(() => api.verify(token), [api, token]);
  return (
    <div className={styles.panel}>
      <p className={styles.panelTitle}>Email confirmation</p>
      {state.status === 'loading' ? <p className={styles.muted}>Confirming</p> : null}
      <RequestOutcome state={state} />
    </div>
  );
}

function LoginForm({api, onLogin}: {api: BenchApi; onLogin: (t: TokenResponse) => void}): ReactNode {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [state, setState] = useState<LoadState<MessageResponse>>({status: 'idle'});

  const onSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setState({status: 'loading'});
    api.login({email: email.trim(), password}).then(onLogin, (error: unknown) =>
      setState(failureState<MessageResponse>(error)),
    );
  };

  return (
    <form className={clsx(styles.panel, styles.stack)} onSubmit={onSubmit}>
      <p className={styles.panelTitle}>Log in</p>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Email</span>
        <input
          className={styles.input}
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </label>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Password</span>
        <input
          className={styles.input}
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </label>
      <button
        type="submit"
        className="button button--primary"
        disabled={state.status === 'loading'}>
        Log in
      </button>
      <RequestOutcome state={state} />
    </form>
  );
}

function SignupForm({api}: {api: BenchApi}): ReactNode {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [affiliation, setAffiliation] = useState('');
  const [state, setState] = useState<LoadState<MessageResponse>>({status: 'idle'});

  const onSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setState({status: 'loading'});
    api
      .signup({
        email: email.trim(),
        password,
        display_name: displayName.trim(),
        ...(affiliation.trim() === '' ? {} : {affiliation: affiliation.trim()}),
      })
      .then(
        (data) => setState({status: 'ok', data}),
        (error: unknown) => setState(failureState<MessageResponse>(error)),
      );
  };

  return (
    <form className={clsx(styles.panel, styles.stack)} onSubmit={onSubmit}>
      <p className={styles.panelTitle}>Sign up</p>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Email</span>
        <input
          className={styles.input}
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
      </label>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Password</span>
        <input
          className={styles.input}
          type="password"
          autoComplete="new-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
      </label>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Display name (public)</span>
        <input
          className={styles.input}
          autoComplete="nickname"
          value={displayName}
          onChange={(e) => setDisplayName(e.target.value)}
          required
        />
      </label>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Affiliation (public, optional)</span>
        <input
          className={styles.input}
          autoComplete="organization"
          value={affiliation}
          onChange={(e) => setAffiliation(e.target.value)}
        />
      </label>
      <button
        type="submit"
        className="button button--secondary"
        disabled={state.status === 'loading'}>
        Sign up
      </button>
      <RequestOutcome state={state} />
    </form>
  );
}

export function MySubmissions({rows}: {rows: SubmissionResult[]}): ReactNode {
  if (rows.length === 0) {
    return <p className={styles.muted}>This account has no submissions.</p>;
  }
  const sorted = [...rows].sort((a, b) => b.submitted_at.localeCompare(a.submitted_at));
  return (
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead>
          <tr>
            <th scope="col">Date</th>
            <th scope="col">Dataset</th>
            <th scope="col">Method</th>
            <th scope="col">Status</th>
            <th scope="col" className={styles.num}>
              Val Pearson
            </th>
            <th scope="col" className={styles.num}>
              Test Pearson
            </th>
            <th scope="col">Flags</th>
            <th scope="col">Rejection reasons</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.submission_id}>
              <td title={row.submitted_at}>{fmtDate(row.submitted_at)}</td>
              <td>
                <code>{row.dataset_slug}</code>
              </td>
              <td>{row.method_name}</td>
              <td>
                <StatusBadge status={row.status} />
              </td>
              <td className={styles.num}>{fmtMetric(row.val?.macro.pearson)}</td>
              <td className={styles.num}>{fmtMetric(row.test?.macro.pearson)}</td>
              <td>
                <FlagChips flags={row.flags} />
              </td>
              <td className={styles.wrap}>
                {row.rejection_reasons.length === 0 ? (
                  <span className={styles.muted}>none</span>
                ) : (
                  <ul className={styles.reasons}>
                    {row.rejection_reasons.map((reason) => (
                      <li key={reason}>{reason}</li>
                    ))}
                  </ul>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function SignedIn({
  api,
  session,
  accessToken,
}: {
  api: BenchApi;
  session: BenchSession;
  accessToken: string;
}): ReactNode {
  const [meState, reloadMe] = useLoad(() => api.me(accessToken), [api, accessToken]);
  const [mineState, reloadMine] = useLoad(() => api.mine(accessToken), [api, accessToken]);

  // A stored token the API no longer accepts is dropped, which returns the page to
  // the logged-out forms.
  const tokenRejected = meState.status === 'error' && meState.httpStatus === 401;
  const {logOut} = session;
  useEffect(() => {
    if (tokenRejected) {
      logOut();
    }
  }, [tokenRejected, logOut]);

  return (
    <>
      <LoadGate state={meState} onRetry={reloadMe}>
        {(me) => (
          <div className={styles.panel}>
            <p className={styles.panelTitle}>{me.display_name}</p>
            <ul className={styles.facts}>
              <li>
                <span className={styles.factLabel}>Email</span>
                {me.email} ({me.email_verified ? 'confirmed' : 'not confirmed'})
              </li>
              <li>
                <span className={styles.factLabel}>Affiliation</span>
                {me.affiliation ?? 'none'}
              </li>
              <li>
                <span className={styles.factLabel}>Session ends</span>
                {session.expiresAt ? fmtDateTime(session.expiresAt) : 'unknown'}
              </li>
            </ul>
            {me.email_verified ? null : (
              <p>
                Submitting requires a confirmed email address. Open the link in the
                confirmation email to confirm it.
              </p>
            )}
            <div className={styles.actions}>
              <Link className="button button--primary button--sm" to="/benchmark/submit">
                Submit predictions
              </Link>
              <Link
                className="button button--secondary button--sm"
                to={`/benchmark/user?id=${encodeURIComponent(me.user_id)}`}>
                Public history
              </Link>
              <button
                type="button"
                className="button button--secondary button--sm"
                onClick={session.logOut}>
                Log out
              </button>
            </div>
          </div>
        )}
      </LoadGate>
      <h2>My submissions</h2>
      <p>
        Every attempt from this account, newest first, including rejected attempts and the
        reasons the grader gave.
      </p>
      <LoadGate state={mineState} onRetry={reloadMine}>
        {(rows) => <MySubmissions rows={rows} />}
      </LoadGate>
    </>
  );
}

function Account(): ReactNode {
  const api = useBenchApi();
  const session = useBenchSession();
  const location = useLocation();
  const verifyToken = new URLSearchParams(location.search).get('verify');

  return (
    <>
      {verifyToken !== null ? <VerifyEmail api={api} token={verifyToken} /> : null}
      {session.accessToken !== null ? (
        <SignedIn api={api} session={session} accessToken={session.accessToken} />
      ) : (
        <div className={styles.columns}>
          <LoginForm api={api} onLogin={session.logIn} />
          <SignupForm api={api} />
        </div>
      )}
    </>
  );
}

export default function AccountApp(): ReactNode {
  return <BenchFrame>{() => <Account />}</BenchFrame>;
}
