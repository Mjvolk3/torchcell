import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {datasetHash, useExplorerUrl} from '@site/src/lib/explorer';
import styles from './DatasetCard.module.css';

type CardProps = {
  /** What was measured, in a few words. */
  title: string;
  /** Benchmark and tc-data slug, e.g. `smf-costanzo2016`. */
  slug: string;
  /** Source publication, e.g. `Costanzo 2016`. */
  citation: string;
  /** Loader class in `torchcell.datasets`. */
  loaderClass: string;
  /** Number of records the loader builds. */
  nRecords: number;
  /** Dataset page in the Sphinx docs. */
  docsUrl?: string;
  /**
   * Site-relative path of the exported draw.io diagram (SVG preferred), e.g.
   * `/img/learn/smf-costanzo2016.svg`. Leave unset until the diagram exists; the
   * card then shows an empty, labeled slot.
   */
  diagramSrc?: string;
  /** Alt text for the diagram. Required once `diagramSrc` is set. */
  diagramAlt?: string;
  /** `CardBlurb`, `CardNotation`, and optionally `CardDetails`, in that order. */
  children: ReactNode;
};

/**
 * The standard dataset card for one dataset: header facts, a diagram slot, a short
 * blurb, and an explainer in the unified notation. Every card has the same parts in
 * the same order, so a reader who has seen one card can read any other.
 */
export default function DatasetCard({
  title,
  slug,
  citation,
  loaderClass,
  nRecords,
  docsUrl,
  diagramSrc,
  diagramAlt,
  children,
}: CardProps): ReactNode {
  const diagramUrl = useBaseUrl(diagramSrc ?? '/');
  const schemaUrl = useExplorerUrl() + datasetHash(loaderClass);
  return (
    <article className={styles.card}>
      <header className={styles.header}>
        <p className={styles.title}>{title}</p>
        <ul className={styles.meta}>
          <li>
            <span className={styles.metaLabel}>Source</span>
            {citation}
          </li>
          <li>
            <span className={styles.metaLabel}>Slug</span>
            <code>{slug}</code>
          </li>
          <li>
            <span className={styles.metaLabel}>Loader</span>
            <code>{loaderClass}</code>
          </li>
          <li>
            <span className={styles.metaLabel}>Records</span>
            {nRecords.toLocaleString('en-US')}
          </li>
          {docsUrl ? (
            <li>
              <Link to={docsUrl}>Dataset page</Link>
            </li>
          ) : null}
          <li>
            {/* A plain anchor: the explorer is a standalone page, not a site route. */}
            <a href={schemaUrl} target="_blank" rel="noopener">
              Schema classes it uses
            </a>
          </li>
        </ul>
      </header>
      <div className={styles.body}>
        <section className={styles.section}>
          <p className={styles.sectionLabel}>Diagram</p>
          {diagramSrc ? (
            <img className={styles.diagram} src={diagramUrl} alt={diagramAlt ?? ''} />
          ) : (
            <div className={styles.diagramEmpty}>
              Diagram slot. Planned: a draw.io schematic of the experiment, exported as SVG.
            </div>
          )}
        </section>
        {children}
      </div>
    </article>
  );
}

type SlotProps = {children: ReactNode};

/** One short paragraph: what was perturbed, what was measured, and how many records. */
export function CardBlurb({children}: SlotProps): ReactNode {
  return (
    <section className={styles.section}>
      <p className={styles.sectionLabel}>What the experiment did</p>
      {children}
    </section>
  );
}

/** The LaTeX explainer, written in the symbols defined on the notation page. */
export function CardNotation({children}: SlotProps): ReactNode {
  return (
    <section className={clsx(styles.section, styles.wide)}>
      <p className={styles.sectionLabel}>In the unified notation</p>
      {children}
    </section>
  );
}

/** Experimental details (strain, medium, replicates, units), each sourced or marked TODO. */
export function CardDetails({children}: SlotProps): ReactNode {
  return (
    <section className={clsx(styles.section, styles.wide)}>
      <p className={styles.sectionLabel}>Experimental details</p>
      {children}
    </section>
  );
}
