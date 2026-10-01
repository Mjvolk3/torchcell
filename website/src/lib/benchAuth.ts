import {useCallback, useState} from 'react';
import type {TokenResponse} from './benchApi';

// The access token lives in sessionStorage: it is dropped when the tab closes and is
// never written to localStorage or a cookie. Call these only in the browser (inside
// <BrowserOnly>); sessionStorage does not exist during server-side rendering.

const STORAGE_KEY = 'torchcell.bench.token';

function readStoredToken(): TokenResponse | null {
  const raw = window.sessionStorage.getItem(STORAGE_KEY);
  if (raw === null) {
    return null;
  }
  const token = JSON.parse(raw) as TokenResponse;
  if (new Date(token.expires_at).getTime() <= Date.now()) {
    window.sessionStorage.removeItem(STORAGE_KEY);
    return null;
  }
  return token;
}

export type BenchSession = {
  /** The bearer token, or null when logged out or expired. */
  accessToken: string | null;
  /** ISO timestamp at which the token stops working, or null when logged out. */
  expiresAt: string | null;
  logIn: (token: TokenResponse) => void;
  logOut: () => void;
};

/** Login state for one benchmark page, backed by sessionStorage. */
export function useBenchSession(): BenchSession {
  const [token, setToken] = useState<TokenResponse | null>(readStoredToken);

  const logIn = useCallback((next: TokenResponse) => {
    window.sessionStorage.setItem(STORAGE_KEY, JSON.stringify(next));
    setToken(next);
  }, []);

  const logOut = useCallback(() => {
    window.sessionStorage.removeItem(STORAGE_KEY);
    setToken(null);
  }, []);

  return {
    accessToken: token ? token.access_token : null,
    expiresAt: token ? token.expires_at : null,
    logIn,
    logOut,
  };
}
