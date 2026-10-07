import React, {lazy, Suspense, type ReactNode} from 'react';
import type {Config, Data, Layout} from 'plotly.js';
import useIsBrowser from '@docusaurus/useIsBrowser';
import {useColorMode} from '@docusaurus/theme-common';
import styles from './bench.module.css';

// Plotly needs `window`, so the bundle is loaded on demand in the browser. The basic
// bundle (scatter, bar, pie) is enough for every chart on this site and is far
// smaller than the full plotly.js.
const PlotlyPlot = lazy(async () => {
  const [plotly, factory] = await Promise.all([
    import('plotly.js-basic-dist-min'),
    import('react-plotly.js/factory'),
  ]);
  return {default: factory.default(plotly.default)};
});

/** Brand palette in series order: orange, red, purple, yellow, blue, gray. */
export const SERIES_COLORS = [
  '#D79B00',
  '#B85450',
  '#9673A6',
  '#D6B656',
  '#6C8EBF',
  '#666666',
] as const;

export type PlotTheme = {
  ink: string;
  muted: string;
  grid: string;
  surface: string;
  /** Neutral color for baselines and reference lines. */
  neutral: string;
};

/** Chart colors for the active color mode, chosen per mode instead of inverted. */
export function usePlotTheme(): PlotTheme {
  const {colorMode} = useColorMode();
  return colorMode === 'dark'
    ? {
        ink: '#e3e3e3',
        muted: '#b0b0ba',
        grid: 'rgba(255, 255, 255, 0.12)',
        surface: '#1b1b1d',
        neutral: '#9a9a9a',
      }
    : {
        ink: '#1c1e21',
        muted: '#5c5c66',
        grid: 'rgba(0, 0, 0, 0.09)',
        surface: '#ffffff',
        neutral: '#666666',
      };
}

/** Shared layout: transparent background, recessive grid, legend below the plot. */
export function baseLayout(theme: PlotTheme): Partial<Layout> {
  const axis = {
    gridcolor: theme.grid,
    linecolor: theme.muted,
    tickcolor: theme.muted,
    zeroline: false,
    automargin: true,
  };
  return {
    autosize: true,
    paper_bgcolor: 'rgba(0, 0, 0, 0)',
    plot_bgcolor: 'rgba(0, 0, 0, 0)',
    font: {
      family: 'system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif',
      size: 12,
      color: theme.ink,
    },
    margin: {l: 56, r: 16, t: 12, b: 44},
    hovermode: 'closest',
    hoverlabel: {
      bgcolor: theme.surface,
      bordercolor: theme.muted,
      font: {color: theme.ink, size: 12},
    },
    showlegend: true,
    legend: {orientation: 'h', x: 0, y: -0.22, font: {color: theme.ink}},
    xaxis: {...axis},
    yaxis: {...axis},
  };
}

const PLOT_CONFIG: Partial<Config> = {
  displayModeBar: false,
  responsive: true,
};

type Props = {
  data: Data[];
  layout: Partial<Layout>;
  /** Accessible name for the chart; the table on the same page carries the values. */
  ariaLabel: string;
};

export default function Plot({data, layout, ariaLabel}: Props): ReactNode {
  // Plotly is never loaded during static generation, even if a chart is rendered
  // outside <BrowserOnly>.
  const isBrowser = useIsBrowser();
  const fallback = <div className={styles.plotFallback}>Loading chart</div>;
  return (
    <div role="img" aria-label={ariaLabel}>
      {isBrowser ? (
        <Suspense fallback={fallback}>
          <PlotlyPlot
            className={styles.plot}
            data={data}
            layout={layout}
            config={PLOT_CONFIG}
            useResizeHandler
          />
        </Suspense>
      ) : (
        fallback
      )}
    </div>
  );
}
