import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

// This file runs in Node.js at build time. The four variables below are read from
// the build environment; see README.md.
//
//   SITE_URL        origin the site is served from
//   BASE_URL        path prefix under that origin; must start and end with "/"
//   BENCH_API_URL   base URL of the benchmark API, including /api/v1
//   BENCH_API_MOCK  "1" loads static/mock/*.json instead of calling the API
//   ONTOLOGY_EXPLORER_URL  the schema explorer the Ontology tab embeds: an absolute
//                   URL, or a path starting with "/" for a copy served by this site
const siteUrl = process.env.SITE_URL ?? 'https://mjvolk3.github.io';
const baseUrl = process.env.BASE_URL ?? '/torchcell/site/';
const benchApiUrl = (
  process.env.BENCH_API_URL ?? 'http://127.0.0.1:8725/api/v1'
).replace(/\/+$/, '');
const benchApiMock = process.env.BENCH_API_MOCK === '1';
const ontologyExplorerUrl =
  process.env.ONTOLOGY_EXPLORER_URL ?? 'https://mjvolk3.github.io/torchcell/ontology/';

// Applies the remembered page width (src/components/WidthToggle.tsx) before the first
// paint, so a reload in wide mode does not flash the default width.
const applyStoredLayout = `(function(){try{if(localStorage.getItem('tc-layout')==='wide'){document.documentElement.setAttribute('data-layout','wide');}}catch(e){}})();`;

const GITHUB_URL = 'https://github.com/Mjvolk3/torchcell';
const SPHINX_URL = 'https://mjvolk3.github.io/torchcell/';

const config: Config = {
  title: 'TorchCell',
  tagline:
    'A Python library and Neo4j knowledge graph for yeast genotype, environment, and phenotype data.',
  // The cell alone; the wordmark is unreadable at favicon size. Written by
  // docs/make_logo.py.
  favicon: 'img/favicon.png',

  url: siteUrl,
  baseUrl,
  trailingSlash: true,

  organizationName: 'Mjvolk3',
  projectName: 'torchcell',

  onBrokenLinks: 'throw',
  onBrokenAnchors: 'throw',
  markdown: {
    hooks: {
      onBrokenMarkdownLinks: 'throw',
      onBrokenMarkdownImages: 'throw',
    },
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  customFields: {
    benchApiUrl,
    benchApiMock,
    ontologyExplorerUrl,
  },

  headTags: [{tagName: 'script', attributes: {}, innerHTML: applyStoredLayout}],

  presets: [
    [
      'classic',
      {
        docs: {
          // Docs are served from the site root so that each navbar tab owns a
          // top-level path: /overview/, /database/, /benchmark/, ...
          routeBasePath: '/',
          sidebarPath: './sidebars.ts',
          sidebarCollapsible: true,
          sidebarCollapsed: true,
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        // Announcements are the blog plugin under another name: one dated Markdown
        // file per announcement in website/announcements/, newest first, with feeds.
        blog: {
          path: 'announcements',
          routeBasePath: 'announcements',
          blogTitle: 'Announcements',
          blogDescription: 'Releases and news from the TorchCell project.',
          blogSidebarTitle: 'All announcements',
          blogSidebarCount: 'ALL',
          showReadingTime: false,
          onUntruncatedBlogPosts: 'ignore',
          feedOptions: {
            type: ['rss', 'atom'],
            title: 'TorchCell announcements',
            description: 'Releases and news from the TorchCell project.',
          },
        },
        theme: {
          customCss: [
            require.resolve('katex/dist/katex.min.css'),
            './src/css/custom.css',
          ],
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/torchcell-logo.png',
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: false,
      },
    },
    navbar: {
      hideOnScroll: false,
      logo: {
        alt: 'TorchCell',
        src: 'img/torchcell-logo.png',
        href: '/',
      },
      items: [
        {type: 'docSidebar', sidebarId: 'overview', label: 'Overview', position: 'left'},
        {type: 'docSidebar', sidebarId: 'database', label: 'Database', position: 'left'},
        {type: 'docSidebar', sidebarId: 'ontology', label: 'Ontology', position: 'left'},
        {type: 'docSidebar', sidebarId: 'docs', label: 'Docs', position: 'left'},
        {type: 'docSidebar', sidebarId: 'benchmark', label: 'Benchmark', position: 'left'},
        {type: 'docSidebar', sidebarId: 'milestones', label: 'Milestones', position: 'left'},
        {type: 'docSidebar', sidebarId: 'tutorials', label: 'Tutorials', position: 'left'},
        {type: 'docSidebar', sidebarId: 'education', label: 'Education', position: 'left'},
        {type: 'docSidebar', sidebarId: 'query', label: 'Query', position: 'left'},
        {to: '/announcements', label: 'Announcements', position: 'right'},
        {type: 'custom-widthToggle', position: 'right'},
        {
          href: GITHUB_URL,
          label: 'GitHub',
          position: 'right',
          className: 'tc-navbar-github',
        },
      ],
    },
    footer: {
      style: 'light',
      links: [
        {
          title: 'Project',
          items: [
            {label: 'GitHub', href: GITHUB_URL},
            {label: 'Issues', href: `${GITHUB_URL}/issues`},
            {label: 'Milestones', href: `${GITHUB_URL}/milestones`},
          ],
        },
        {
          title: 'Documentation',
          items: [
            {label: 'Sphinx docs', href: SPHINX_URL},
            {label: 'Datasets', href: `${SPHINX_URL}datasets/index.html`},
            {label: 'Ontology explorer', href: `${SPHINX_URL}ontology/`},
          ],
        },
        {
          title: 'Database',
          items: [
            {
              label: 'Neo4j Browser',
              href: 'https://torchcell-database.ncsa.illinois.edu:7473/browser/',
            },
            {
              label: 'Releases and compatibility',
              href: `${SPHINX_URL}database/compatibility.html`,
            },
          ],
        },
      ],
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'cypher', 'csv'],
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
