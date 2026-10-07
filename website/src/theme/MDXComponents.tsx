import MDXComponents from '@theme-original/MDXComponents';
import StatusBadge from '@site/src/components/StatusBadge';
import Todo from '@site/src/components/Todo';
import OntologyExplorer from '@site/src/components/OntologyExplorer';
import {LinkCard, LinkGrid} from '@site/src/components/LinkCard';
import DatasetCard, {
  CardBlurb,
  CardDetails,
  CardNotation,
} from '@site/src/components/DatasetCard';

// Components listed here are available in every .mdx file without an import.
export default {
  ...MDXComponents,
  StatusBadge,
  Todo,
  OntologyExplorer,
  LinkCard,
  LinkGrid,
  DatasetCard,
  CardBlurb,
  CardNotation,
  CardDetails,
};
