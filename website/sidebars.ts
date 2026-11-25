import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// One sidebar per navbar tab, all with the same shape: the tab's landing page first,
// then named groups of pages. Groups are headings, not dropdowns (sidebarCollapsible is
// false in docusaurus.config.ts), so every page of a tab is visible at once and every
// entry in the sidebar is a link that opens a page.
// The one exception is the organism groups under Benchmark > Datasets: they are
// dropdowns that start closed, so a long list of tasks does not bury the top-level pages.

const SPHINX_URL = 'https://mjvolk3.github.io/torchcell/';
const GITHUB_URL = 'https://github.com/Mjvolk3/torchcell';

const sidebars: SidebarsConfig = {
  overview: [
    'overview/index',
    {
      type: 'category',
      label: 'Concepts',
      items: ['overview/data-model', 'overview/site-map'],
    },
    {
      type: 'category',
      label: 'Project links',
      items: [
        {type: 'link', label: 'GitHub repository', href: GITHUB_URL},
        {type: 'link', label: 'Sphinx documentation', href: SPHINX_URL},
        {type: 'link', label: 'Issues', href: `${GITHUB_URL}/issues`},
      ],
    },
  ],

  database: [
    'database/index',
    {
      type: 'category',
      label: 'Access',
      items: ['database/browser', 'database/bolt', 'database/query'],
    },
    {
      type: 'category',
      label: 'Releases',
      items: ['database/releases'],
    },
  ],

  ontology: [
    'ontology/index',
    {
      type: 'category',
      label: 'Schema explorer',
      items: ['ontology/explorer'],
    },
    {
      type: 'category',
      label: 'Related',
      items: ['ontology/learn'],
    },
  ],

  docs: [
    'docs/index',
    {
      type: 'category',
      label: 'Sphinx documentation',
      items: [
        {type: 'link', label: 'Guide', href: `${SPHINX_URL}guide/index.html`},
        {type: 'link', label: 'Datasets', href: `${SPHINX_URL}datasets/index.html`},
        {type: 'link', label: 'API reference', href: SPHINX_URL},
      ],
    },
    {
      type: 'category',
      label: 'Database pages',
      items: [
        {
          type: 'link',
          label: 'Releases and compatibility',
          href: `${SPHINX_URL}database/compatibility.html`,
        },
        {
          type: 'link',
          label: 'Dataset downloads',
          href: `${SPHINX_URL}guide/downloads.html`,
        },
      ],
    },
  ],

  benchmark: [
    'benchmark/index',
    {
      type: 'category',
      label: 'Boards',
      items: ['benchmark/leaderboard', 'benchmark/submit', 'benchmark/account', 'benchmark/user'],
    },
    {
      type: 'category',
      label: 'Datasets',
      items: [
        'benchmark/datasets/index',
        {
          type: 'category',
          label: 'Saccharomyces cerevisiae',
          collapsible: true,
          collapsed: true,
          items: ['benchmark/datasets/gene-essentiality-sgd'],
        },
        {
          type: 'category',
          label: 'Escherichia coli',
          collapsible: true,
          collapsed: true,
          items: [
            'benchmark/datasets/ecoli/index',
            'benchmark/datasets/ecoli/growth-rate-campos2018',
            'benchmark/datasets/ecoli/morphology-campos2018',
            'benchmark/datasets/ecoli/gene-interaction-babu2014',
            'benchmark/datasets/ecoli/metabolome-fuhrer2017',
            'benchmark/datasets/ecoli/processing',
          ],
        },
      ],
    },
    {
      type: 'category',
      label: 'Baselines and metrics',
      items: ['benchmark/baselines', 'benchmark/encodings', 'benchmark/metrics'],
    },
    {
      type: 'category',
      label: 'Submitting',
      items: [
        'benchmark/submission-format',
        'benchmark/training-protocol',
        'benchmark/limits',
        'benchmark/github-ingestion',
      ],
    },
    {
      type: 'category',
      label: 'Integrity',
      items: ['benchmark/status', 'benchmark/integrity'],
    },
  ],

  milestones: [
    'milestones/index',
    {
      type: 'category',
      label: 'On GitHub',
      items: [
        {type: 'link', label: 'Milestones', href: `${GITHUB_URL}/milestones`},
        {type: 'link', label: 'Issues', href: `${GITHUB_URL}/issues`},
      ],
    },
  ],

  tutorials: [
    'tutorials/index',
    {
      type: 'category',
      label: 'Working with datasets',
      items: [
        'tutorials/download-and-benchmark',
        'tutorials/subsetting-with-indices',
        'tutorials/gene-embeddings',
      ],
    },
    {
      type: 'category',
      label: 'Graphs and models',
      items: [
        'tutorials/graphs-and-perturbations',
        'tutorials/training-models',
        'tutorials/simple-benchmarks',
      ],
    },
    {
      type: 'category',
      label: 'Querying new datasets',
      items: ['tutorials/querying-a-new-dataset', 'tutorials/analysis-before-conversion'],
    },
    {
      type: 'category',
      label: 'Worked example',
      items: ['tutorials/cgt-worked-example'],
    },
  ],

  learn: [
    'learn/index',
    'learn/notation',
    {
      type: 'category',
      label: 'Dataset cards',
      items: [
        'learn/cards/smf-costanzo2016',
        'learn/cards/amino-acid-mulleder2016',
        'learn/cards/betaxanthin-cachera2023',
      ],
    },
  ],

  query: [
    'query/index',
    {
      type: 'category',
      label: 'Query modes (planned)',
      items: ['query/chemical-similarity', 'query/gene-similarity', 'query/homologs'],
    },
  ],
};

export default sidebars;
