import React, {type ReactNode} from 'react';
import {useExplorerUrl} from '@site/src/lib/explorer';

type Props = {
  /** Optional hash to open on, from `classHash` or `datasetHash` in `src/lib/explorer`. */
  hash?: string;
};

/**
 * The schema explorer in a frame that fills the window below the navbar, with a link
 * that opens it alone in a new tab. The explorer re-fits its map whenever the frame
 * changes size, so it follows the page-width toggle.
 */
export default function OntologyExplorer({hash = ''}: Props): ReactNode {
  const url = useExplorerUrl() + hash;
  return (
    <div className="tc-explorer">
      <p className="tc-explorer__bar">
        <a href={url} target="_blank" rel="noopener">
          Open the explorer in its own tab
        </a>
      </p>
      <iframe className="tc-explorer__frame" src={url} title="TorchCell schema explorer" />
    </div>
  );
}
