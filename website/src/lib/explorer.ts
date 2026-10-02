import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';

/**
 * URL of the schema explorer, without a hash. `ONTOLOGY_EXPLORER_URL` sets it at build
 * time (see docusaurus.config.ts): an absolute URL, or a path that starts with `/` for
 * a copy served by this site, which is resolved against the site's base URL.
 */
export function useExplorerUrl(): string {
  const {siteConfig} = useDocusaurusContext();
  const configured = String(siteConfig.customFields?.ontologyExplorerUrl);
  const local = configured.startsWith('/');
  const resolved = useBaseUrl(local ? configured : '/');
  return local ? resolved : configured;
}

/** Explorer link that opens on one schema class, e.g. `FitnessPhenotype`. */
export function classHash(className: string): string {
  return `#${encodeURIComponent(className)}`;
}

/** Explorer link that opens on one served dataset and lights up the classes it uses. */
export function datasetHash(loaderClass: string): string {
  return `#dataset=${encodeURIComponent(loaderClass)}`;
}
