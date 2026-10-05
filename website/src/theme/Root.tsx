import React, {type ReactNode} from 'react';
import {useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Wraps the whole site. It adds the two bars that say a build is not the public one:
// the mock-data bar on every page under /benchmark/ when the site is built with
// BENCH_API_MOCK=1, and the staging bar on every page when it is built with
// SITE_ENV=staging.
export default function Root({children}: {children: ReactNode}): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const {pathname} = useLocation();
  const benchmarkRoot = useBaseUrl('/benchmark');
  const mock = siteConfig.customFields?.benchApiMock === true;
  const staging = siteConfig.customFields?.siteEnv === 'staging';
  const onBenchmarkPage = pathname.startsWith(benchmarkRoot);
  return (
    <>
      {children}
      {mock && onBenchmarkPage ? (
        <div className="tc-mock-bar" role="status">
          MOCK DATA, not real results
        </div>
      ) : null}
      {staging && !(mock && onBenchmarkPage) ? (
        <div className="tc-mock-bar tc-staging-bar" role="status">
          STAGING: test accounts and test submissions, not the public board
        </div>
      ) : null}
    </>
  );
}
