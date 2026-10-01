import React, {type ReactNode} from 'react';
import {useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';

// Wraps the whole site. Its only job is the mock-data bar: when the site is built with
// BENCH_API_MOCK=1, every page under /benchmark/ carries a fixed bar saying so.
export default function Root({children}: {children: ReactNode}): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const {pathname} = useLocation();
  const benchmarkRoot = useBaseUrl('/benchmark');
  const mock = siteConfig.customFields?.benchApiMock === true;
  const onBenchmarkPage = pathname.startsWith(benchmarkRoot);
  return (
    <>
      {children}
      {mock && onBenchmarkPage ? (
        <div className="tc-mock-bar" role="status">
          MOCK DATA, not real results
        </div>
      ) : null}
    </>
  );
}
