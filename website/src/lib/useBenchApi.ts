import {useMemo} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {createBenchApi, type BenchApi} from './benchApi';

type BenchCustomFields = {
  benchApiUrl: string;
  benchApiMock: boolean;
};

/** The build-time benchmark settings from `customFields` in docusaurus.config.ts. */
export function useBenchConfig(): BenchCustomFields {
  const {siteConfig} = useDocusaurusContext();
  return siteConfig.customFields as BenchCustomFields;
}

/** The benchmark API client for this build: HTTP by default, fixtures in mock mode. */
export function useBenchApi(): BenchApi {
  const {benchApiUrl, benchApiMock} = useBenchConfig();
  const mockBaseUrl = useBaseUrl('/mock/');
  return useMemo(
    () => createBenchApi({baseUrl: benchApiUrl, mock: benchApiMock, mockBaseUrl}),
    [benchApiUrl, benchApiMock, mockBaseUrl],
  );
}
