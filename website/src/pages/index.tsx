import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import Layout from '@theme/Layout';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {LinkCard, LinkGrid} from '@site/src/components/LinkCard';
import {formatCount, useReleaseCounts} from '@site/src/components/ReleaseCounts';
import styles from './index.module.css';

const GITHUB_URL = 'https://github.com/Mjvolk3/torchcell';

// One card per navbar tab, in navbar order.
const TABS: {title: string; href: string; body: string}[] = [
  {
    title: 'Overview',
    href: '/overview/',
    body: 'What TorchCell is, the typed experiment record, and where each part lives.',
  },
  {
    title: 'Benchmark',
    href: '/benchmark/',
    body: 'Fixed splits, standard baselines, and a board per dataset. Upload predictions; the server scores them.',
  },
  {
    title: 'Ontology',
    href: '/ontology/',
    body: 'The schema of experiment records, as an interactive explorer.',
  },
  {
    title: 'Database',
    href: '/database/',
    body: 'The served Neo4j knowledge graph: browser, Bolt URI, releases.',
  },
  {
    title: 'Query',
    href: '/query/',
    body: 'Planned shortcuts for querying by chemical, gene, or homolog similarity.',
  },
  {
    title: 'Docs',
    href: '/docs/',
    body: 'Links to the Sphinx guide, the dataset pages, and the API reference.',
  },
  {
    title: 'Tutorials',
    href: '/tutorials/',
    body: 'Nine planned notebooks, most of them about datasets.',
  },
  {
    title: 'Learn',
    href: '/learn/',
    body: 'One standard card per dataset: a diagram, a blurb, and a shared notation.',
  },
  {
    title: 'Milestones',
    href: '/milestones/',
    body: 'Planned work, tracked as GitHub milestones.',
  },
];

export default function Home(): ReactNode {
  const {siteConfig} = useDocusaurusContext();
  const counts = useReleaseCounts();
  return (
    <Layout title="Home" description={siteConfig.tagline}>
      <header className={styles.hero}>
        <div className="container">
          <h1 className={clsx(styles.title, 'tc-gradient-text')}>{siteConfig.title}</h1>
          <p className={styles.tagline}>{siteConfig.tagline}</p>
          <div className={styles.actions}>
            <Link className="button button--primary button--lg" to="/overview/">
              Overview
            </Link>
            <Link className="button button--secondary button--lg" to="/benchmark/leaderboard">
              Leaderboard
            </Link>
            <Link className="button button--secondary button--lg" to={GITHUB_URL}>
              GitHub
            </Link>
          </div>
          <ul className={styles.stats}>
            <li className={styles.stat}>
              <span className={styles.statValue}>{counts.nDatasets}</span>
              <span className={styles.statLabel}>datasets in the served knowledge graph</span>
            </li>
            <li className={styles.stat}>
              <span className={styles.statValue}>{formatCount(counts.nExperiments)}</span>
              <span className={styles.statLabel}>experiment records</span>
            </li>
            {counts.nGenotypeEnvironmentPairs !== null && (
              <li className={styles.stat}>
                <span className={styles.statValue}>
                  {formatCount(counts.nGenotypeEnvironmentPairs)}
                </span>
                <span className={styles.statLabel}>
                  distinct genotype-environment combinations
                </span>
              </li>
            )}
            <li className={styles.stat}>
              <span className={styles.statValue}>{counts.releaseDate}</span>
              <span className={styles.statLabel}>database release these counts refer to</span>
            </li>
          </ul>
        </div>
      </header>
      <main className={clsx('container', styles.section)}>
        <div className={styles.record}>
          <h2 className={styles.sectionTitle}>One record type for every measurement</h2>
          <p>
            TorchCell records each published measurement as a typed experiment, a pydantic{' '}
            <code>genotype x environment -&gt; phenotype</code> object. Provenance is part of
            the data model: source files are pinned by sha256, extracted values carry the
            verbatim source quote, and built datasets are checked at verification levels L0
            to L4.
          </p>
        </div>
        <h2 className={styles.sectionTitle}>Sections</h2>
        <p className={styles.sectionLead}>
          Each tab in the navigation bar opens one section with its own collapsible sidebar.
        </p>
        <LinkGrid>
          {TABS.map((tab) => (
            <LinkCard key={tab.title} title={tab.title} href={tab.href}>
              {tab.body}
            </LinkCard>
          ))}
        </LinkGrid>
      </main>
    </Layout>
  );
}
