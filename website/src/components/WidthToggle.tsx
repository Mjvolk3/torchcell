import React, {useEffect, useState, type ReactNode} from 'react';
import {createStorageSlot} from '@docusaurus/theme-common';

// The page has two widths. The default keeps the content column at the theme's reading
// width. "Wide" lets it take the whole window, which is what a table, a chart, or the
// schema explorer wants on a large screen. The choice is remembered per browser under
// this key; an inline script in the page head (docusaurus.config.ts) applies it before
// the first paint, so a reload does not flash the default width.
export const LAYOUT_STORAGE_KEY = 'tc-layout';
const storage = createStorageSlot(LAYOUT_STORAGE_KEY);

type Layout = 'default' | 'wide';

function applyLayout(layout: Layout): void {
  if (layout === 'wide') {
    document.documentElement.setAttribute('data-layout', 'wide');
  } else {
    document.documentElement.removeAttribute('data-layout');
  }
}

type Props = {
  /** True when the navbar renders its items inside the mobile sidebar. */
  mobile?: boolean;
};

/** Navbar button that switches the page between the default and the full width. */
export default function WidthToggle({mobile}: Props): ReactNode {
  const [layout, setLayout] = useState<Layout>('default');

  // The head script has already set the attribute; read it once the page is live.
  useEffect(() => {
    const current = document.documentElement.getAttribute('data-layout');
    setLayout(current === 'wide' ? 'wide' : 'default');
  }, []);

  // On a phone the content already spans the screen, so there is nothing to toggle.
  if (mobile) {
    return null;
  }

  const wide = layout === 'wide';
  const label = wide ? 'Return to the default page width' : 'Fit the page to the screen width';
  const toggle = (): void => {
    const next: Layout = wide ? 'default' : 'wide';
    setLayout(next);
    applyLayout(next);
    storage.set(next);
  };

  return (
    <button
      type="button"
      className="clean-btn tc-width-toggle"
      aria-pressed={wide}
      aria-label={label}
      title={label}
      onClick={toggle}>
      <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
        {wide ? (
          // Arrows pointing in: back to the reading width.
          <path
            d="M3 5v14M21 5v14M7 12h3M10 12l-2.5-2.5M10 12l-2.5 2.5M17 12h-3M14 12l2.5-2.5M14 12l2.5 2.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        ) : (
          // Arrows pointing out: take the whole window.
          <path
            d="M3 5v14M21 5v14M6.5 12h11M6.5 12l2.5-2.5M6.5 12l2.5 2.5M17.5 12l-2.5-2.5M17.5 12l-2.5 2.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        )}
      </svg>
    </button>
  );
}
