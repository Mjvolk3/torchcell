import React, {useEffect, useState, type FormEvent, type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {
  LOGIN_ERRORS,
  isLoginErrorCode,
  type ApiToken,
  type ApiTokenCreated,
  type BenchApi,
  type SubmissionResult,
  type TokenResponse,
  type UserPrivate,
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

/** What the API put in the URL fragment when it sent the browser back here. */
type LoginReturn = {code: string | null; error: string | null};

// A sign-in ends with a full page load of this page carrying `#login_code=...` or
// `#login_error=...`. The fragment is read once per page load and removed from the
// address bar at once, so a reload or a shared link never replays it. The exchange is
// kept at module level for the same reason: a code works once, and the component may
// mount more than once.
let loginReturn: LoginReturn | undefined;
let exchangeRequest: Promise<TokenResponse> | undefined;

function takeLoginReturn(): LoginReturn {
  if (loginReturn === undefined) {
    const fragment = new URLSearchParams(window.location.hash.replace(/^#/, ''));
    loginReturn = {code: fragment.get('login_code'), error: fragment.get('login_error')};
    if (loginReturn.code !== null || loginReturn.error !== null) {
      const {pathname, search} = window.location;
      window.history.replaceState(window.history.state, '', `${pathname}${search}`);
    }
  }
  return loginReturn;
}

/** Finishes a sign-in: trades the one-time code for the session token. */
function CompleteSignIn({
  api,
  code,
  onSignedIn,
}: {
  api: BenchApi;
  code: string;
  onSignedIn: (token: TokenResponse) => void;
}): ReactNode {
  const [state, setState] = useState<LoadState<TokenResponse>>({status: 'loading'});

  useEffect(() => {
    let cancelled = false;
    exchangeRequest ??= api.exchange(code);
    exchangeRequest.then(
      (token) => {
        if (!cancelled) {
          onSignedIn(token);
        }
      },
      (error: unknown) => {
        if (!cancelled) {
          setState(failureState<TokenResponse>(error));
        }
      },
    );
    return () => {
      cancelled = true;
    };
  }, [api, code, onSignedIn]);

  switch (state.status) {
    case 'unreachable':
      return <ApiUnreachable url={state.url} />;
    case 'error':
      return (
        <ErrorNotice
          message="The sign-in could not be completed. Start it again below."
          reasons={[state.message, ...state.reasons]}
        />
      );
    default:
      return <p className={styles.muted}>Completing sign-in</p>;
  }
}

function SignInPanel({
  api,
  onSignedIn,
}: {
  api: BenchApi;
  onSignedIn: (token: TokenResponse) => void;
}): ReactNode {
  const loginUrl = api.loginUrl();
  return (
    <div className={clsx(styles.panel, styles.stack)}>
      <p className={styles.panelTitle}>Sign in</p>
      <p>
        Sign-in is handled by CILogon. Choose your university or institute from its
        list; ORCID, GitHub, Google and Microsoft are there too. The first sign-in opens
        your account, and there is no password to set here.
      </p>
      {loginUrl === null ? (
        <button
          type="button"
          className="button button--primary"
          onClick={() => {
            void api.exchange('mock').then(onSignedIn);
          }}>
          Sign in (mock session)
        </button>
      ) : (
        // A plain link: the browser has to leave the site for CILogon.
        <a className="button button--primary" href={loginUrl}>
          Sign in with CILogon
        </a>
      )}
    </div>
  );
}

function ProfileForm({
  api,
  accessToken,
  me,
  onSaved,
}: {
  api: BenchApi;
  accessToken: string;
  me: UserPrivate;
  onSaved: (saved: UserPrivate) => void;
}): ReactNode {
  const [displayName, setDisplayName] = useState(me.display_name);
  const [affiliation, setAffiliation] = useState(me.affiliation ?? '');
  const [state, setState] = useState<LoadState<UserPrivate>>({status: 'idle'});

  const onSubmit = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setState({status: 'loading'});
    api
      .updateProfile(accessToken, {
        display_name: displayName.trim(),
        affiliation: affiliation.trim() === '' ? null : affiliation.trim(),
      })
      .then(
        (data) => {
          setState({status: 'ok', data});
          onSaved(data);
        },
        (error: unknown) => setState(failureState<UserPrivate>(error)),
      );
  };

  return (
    <form className={clsx(styles.panel, styles.stack)} onSubmit={onSubmit}>
      <p className={styles.panelTitle}>Public profile</p>
      <label className={styles.field}>
        <span className={styles.fieldLabel}>Display name (public)</span>
        <input
          className={styles.input}
          autoComplete="nickname"
          minLength={2}
          maxLength={60}
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
          maxLength={120}
          value={affiliation}
          onChange={(e) => setAffiliation(e.target.value)}
        />
      </label>
      <button
        type="submit"
        className="button button--secondary"
        disabled={state.status === 'loading'}>
        Save profile
      </button>
      {state.status === 'ok' ? (
        <div className={clsx(styles.notice, styles.noticeOk)} role="status">
          Profile saved.
        </div>
      ) : null}
      {state.status === 'unreachable' ? <ApiUnreachable url={state.url} /> : null}
      {state.status === 'error' ? (
        <ErrorNotice message={state.message} reasons={state.reasons} />
      ) : null}
    </form>
  );
}

/**
 * Personal API tokens: what a script sends instead of a browser session. A token is
 * shown once, in the answer to its creation, and only its first characters afterwards.
 */
function ApiTokens({api, accessToken}: {api: BenchApi; accessToken: string}): ReactNode {
  const [tokensState, reloadTokens] = useLoad(() => api.tokens(accessToken), [api, accessToken]);
  const [name, setName] = useState('');
  const [created, setCreated] = useState<LoadState<ApiTokenCreated>>({status: 'idle'});
  const [revoked, setRevoked] = useState<LoadState<string>>({status: 'idle'});

  const onCreate = (event: FormEvent<HTMLFormElement>): void => {
    event.preventDefault();
    setCreated({status: 'loading'});
    api.createToken(accessToken, name.trim()).then(
      (data) => {
        setCreated({status: 'ok', data});
        setName('');
        reloadTokens();
      },
      (error: unknown) => setCreated(failureState<ApiTokenCreated>(error)),
    );
  };

  const onRevoke = (token: ApiToken): void => {
    setRevoked({status: 'loading'});
    api.revokeToken(accessToken, token.token_id).then(
      () => {
        setRevoked({status: 'ok', data: token.name});
        // The token shown once must not stay on screen after it stops working.
        setCreated((state) =>
          state.status === 'ok' && state.data.token_id === token.token_id
            ? {status: 'idle'}
            : state,
        );
        reloadTokens();
      },
      (error: unknown) => setRevoked(failureState<string>(error)),
    );
  };

  return (
    <>
      <h2 id="api-tokens">API tokens</h2>
      <p>
        A token lets a script submit for this account without a browser, through the same
        endpoint the submit form uses. Set it as <code>TC_BENCH_TOKEN</code>; the{' '}
        <Link to="/benchmark/submit">submit page</Link> has the setup. A token can read
        this account and submit; it cannot edit the profile or create other tokens. The
        submission quota is the account's, however many tokens it holds.
      </p>
      {created.status === 'ok' ? (
        <div className={clsx(styles.notice, styles.noticeOk)} role="status">
          <p>
            Token <strong>{created.data.name}</strong> created. Copy it now: it is not stored
            and cannot be shown again.
          </p>
          <pre className={styles.secret}>
            <code>{created.data.token}</code>
          </pre>
        </div>
      ) : null}
      {created.status === 'error' ? (
        <ErrorNotice message={created.message} reasons={created.reasons} />
      ) : null}
      {created.status === 'unreachable' ? <ApiUnreachable url={created.url} /> : null}
      {revoked.status === 'ok' ? (
        <div className={clsx(styles.notice, styles.noticeOk)} role="status">
          Token {revoked.data} revoked. It no longer works.
        </div>
      ) : null}
      {revoked.status === 'error' ? (
        <ErrorNotice message={revoked.message} reasons={revoked.reasons} />
      ) : null}
      {revoked.status === 'unreachable' ? <ApiUnreachable url={revoked.url} /> : null}
      <LoadGate state={tokensState} onRetry={reloadTokens}>
        {(tokens) =>
          tokens.length === 0 ? (
            <p className={styles.muted}>This account has no API tokens.</p>
          ) : (
            <div className={styles.tableWrap}>
              <table className={styles.table}>
                <thead>
                  <tr>
                    <th scope="col">Name</th>
                    <th scope="col">Token</th>
                    <th scope="col">Created</th>
                    <th scope="col">Last used</th>
                    <th scope="col">
                      <span className={styles.srOnly}>Actions</span>
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {tokens.map((token) => (
                    <tr key={token.token_id}>
                      <td>{token.name}</td>
                      <td>
                        <code>{token.hint}…</code>
                      </td>
                      <td title={token.created_at}>{fmtDate(token.created_at)}</td>
                      <td title={token.last_used_at ?? undefined}>
                        {token.last_used_at ? fmtDate(token.last_used_at) : 'never'}
                      </td>
                      <td>
                        <button
                          type="button"
                          className="button button--secondary button--sm"
                          disabled={revoked.status === 'loading'}
                          onClick={() => onRevoke(token)}>
                          Revoke
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        }
      </LoadGate>
      <form className={styles.inlineForm} onSubmit={onCreate}>
        <label className={styles.field}>
          <span className={styles.fieldLabel}>Name of a new token</span>
          <input
            className={styles.input}
            maxLength={60}
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="for example: laptop, cluster"
            required
          />
        </label>
        <button
          type="submit"
          className="button button--primary"
          disabled={created.status === 'loading'}>
          Create token
        </button>
      </form>
    </>
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
  // The profile as last saved from this page. Showing it directly, rather than loading
  // the account again, keeps the form and its confirmation on screen.
  const [savedProfile, setSavedProfile] = useState<UserPrivate | null>(null);

  // A stored token the API no longer accepts is dropped, which returns the page to
  // the sign-in panel.
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
        {(loaded) => {
          const me = savedProfile ?? loaded;
          return (
            <div className={styles.columns}>
              <div className={styles.panel}>
                <p className={styles.panelTitle}>{me.display_name}</p>
                <ul className={styles.facts}>
                  <li>
                    <span className={styles.factLabel}>Email</span>
                    {me.email}
                  </li>
                  <li>
                    <span className={styles.factLabel}>Signed in through</span>
                    {me.identity_provider ?? 'CILogon'}
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
                {me.approved ? null : (
                  <p>
                    This account is waiting for approval by a maintainer. You can submit
                    once it is approved.
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
                    Sign out
                  </button>
                </div>
              </div>
              <ProfileForm
                api={api}
                accessToken={accessToken}
                me={me}
                onSaved={setSavedProfile}
              />
            </div>
          );
        }}
      </LoadGate>
      <ApiTokens api={api} accessToken={accessToken} />
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
  const [returned] = useState(takeLoginReturn);

  if (session.accessToken !== null) {
    return <SignedIn api={api} session={session} accessToken={session.accessToken} />;
  }
  return (
    <>
      {returned.error !== null ? (
        <ErrorNotice
          message={
            isLoginErrorCode(returned.error)
              ? LOGIN_ERRORS[returned.error]
              : LOGIN_ERRORS.failed
          }
        />
      ) : null}
      {returned.code !== null ? (
        <CompleteSignIn api={api} code={returned.code} onSignedIn={session.logIn} />
      ) : null}
      <SignInPanel api={api} onSignedIn={session.logIn} />
    </>
  );
}

export default function AccountApp(): ReactNode {
  return <BenchFrame>{() => <Account />}</BenchFrame>;
}
