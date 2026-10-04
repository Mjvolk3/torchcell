import React, {type ReactNode} from 'react';
import CodeBlock from '@theme/CodeBlock';
import {useBenchConfig} from '@site/src/lib/useBenchApi';

/** The placeholder a code sample writes where the API base URL of this build goes. */
export const API_PLACEHOLDER = '{API}';

/**
 * A code sample that names the benchmark API. `{API}` in the text is replaced by the
 * base URL this site was built with (BENCH_API_URL), so a sample can be copied and run
 * against the API the page itself talks to.
 */
export default function BenchCode({
  language,
  title,
  children,
}: {
  language: string;
  title?: string;
  children: string;
}): ReactNode {
  const {benchApiUrl} = useBenchConfig();
  return (
    <CodeBlock language={language} title={title}>
      {children.trim().replaceAll(API_PLACEHOLDER, benchApiUrl)}
    </CodeBlock>
  );
}
