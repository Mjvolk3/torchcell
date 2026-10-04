import type {SidebarsConfig} from '@docusaurus/plugin-content-docs';

// One sidebar per navbar tab, all with the same shape: the tab's landing page first,
// then named groups of pages. Groups are headings, not dropdowns (sidebarCollapsible is
// false in docusaurus.config.ts), so every page of a tab is visible at once and every
// entry in the sidebar is a link that opens a page.

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
      items: ['database/browser', 'database/bolt'],
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
      items: ['ontology/education'],
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
      items: ['benchmark/leaderboard', 'benchmark/submit', 'benchmark/account'],
    },
    {
      type: 'category',
      label: 'Datasets and baselines',
      items: [
        'benchmark/datasets',
        'benchmark/baselines',
        'benchmark/encodings',
        'benchmark/metrics',
      ],
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

  education: [
    'education/index',
    'education/notation',
    {
      type: 'category',
      label: 'Dataset cards',
      items: [
        'education/cards/smf-costanzo2016',
        'education/cards/amino-acid-mulleder2016',
        'education/cards/betaxanthin-cachera2023',
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
