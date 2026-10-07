import React, {type ReactNode} from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import type {ReleaseCounts as Counts} from '@site/docusaurus.config';

/** The counts of the newest committed release snapshot (docusaurus.config.ts). */
export function useReleaseCounts(): Counts {
  const {siteConfig} = useDocusaurusContext();
  return siteConfig.customFields?.release as Counts;
}

/** 52743047 -> "52.7 million"; 4313 -> "4,313". */
export function formatCount(n: number): string {
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)} million`;
  }
  return n.toLocaleString('en-US');
}

type Field = 'releaseDate' | 'nDatasets' | 'nExperiments' | 'nGenotypeEnvironmentPairs';

/**
 * One release count, inline, for MDX: `<ReleaseCount field="nExperiments" />` renders
 * "52.7 million". The pairs field renders "not yet counted" until `releases
 * count-pairs` has filled the snapshot, so a page never shows a stale or invented
 * number.
 */
export default function ReleaseCount({field}: {field: Field}): ReactNode {
  const counts = useReleaseCounts();
  if (field === 'releaseDate') {
    return <>{counts.releaseDate}</>;
  }
  const value = counts[field];
  if (value === null) {
    return <>not yet counted</>;
  }
  return <>{formatCount(value)}</>;
}
