import React, {useCallback, useEffect, useRef, useState, type ReactNode} from 'react';
import {useLocation} from '@docusaurus/router';
import {createStorageSlot} from '@docusaurus/theme-common';

// A handle on the right edge of the docs sidebar. Drag it to set the sidebar's width;
// double-click it to fit the width to the longest visible entry, so no label wraps, and
// double-click again to return to the default. The width is remembered per browser under
// this key; an inline script in the page head (docusaurus.config.ts) applies it before
// the first paint.
export const SIDEBAR_WIDTH_KEY = 'tc-sidebar-width';
const storage = createStorageSlot(SIDEBAR_WIDTH_KEY);

const SIDEBAR = '.theme-doc-sidebar-container';
const LINKS = '.theme-doc-sidebar-menu .menu__link';
const MIN_WIDTH = 180;
const DESKTOP = 997; // the theme shows the docs sidebar from this window width up
const ARROW_ROOM = 36; // the open/close arrow right of a group's label
const EDGE_ROOM = 28; // the sidebar's own right padding, its border and a scrollbar

function maxWidth(): number {
  return Math.round(window.innerWidth * 0.6);
}

function setWidth(pixels: number | null): void {
  const root = document.documentElement;
  if (pixels === null) {
    root.style.removeProperty('--doc-sidebar-width');
    storage.del();
    return;
  }
  const width = Math.min(Math.max(Math.round(pixels), MIN_WIDTH), maxWidth());
  root.style.setProperty('--doc-sidebar-width', `${width}px`);
  storage.set(String(width));
}

/** Width of a string in the font an element renders its text in. */
function textWidth(context: CanvasRenderingContext2D, element: HTMLElement): number {
  const style = window.getComputedStyle(element);
  let text = (element.textContent ?? '').trim();
  if (style.textTransform === 'uppercase') {
    text = text.toUpperCase();
  }
  context.font = `${style.fontStyle} ${style.fontWeight} ${style.fontSize} ${style.fontFamily}`;
  const spacing = Number.parseFloat(style.letterSpacing) || 0;
  return context.measureText(text).width + spacing * text.length;
}

/**
 * The narrowest sidebar in which no visible entry wraps.
 *
 * Each label is measured as text, in its own font, on a canvas. Measuring the laid-out
 * element does not work: the theme clips a label inside its own box, so the element
 * never reports a width wider than the sidebar it is in.
 */
function fitWidth(sidebar: Element): number {
  const context = document.createElement('canvas').getContext('2d');
  if (!context) {
    return MIN_WIDTH;
  }
  const left = sidebar.getBoundingClientRect().left;
  let widest = MIN_WIDTH;
  sidebar.querySelectorAll<HTMLElement>(LINKS).forEach((link) => {
    if (link.offsetParent === null) {
      return; // inside a closed group
    }
    const style = window.getComputedStyle(link);
    const padding = Number.parseFloat(style.paddingLeft) + Number.parseFloat(style.paddingRight);
    const arrow = link.closest('.menu__list-item-collapsible') ? ARROW_ROOM : 0;
    const indent = link.getBoundingClientRect().left - left;
    widest = Math.max(widest, indent + padding + textWidth(context, link) + arrow + EDGE_ROOM);
  });
  return Math.ceil(widest);
}

export default function SidebarResizer(): ReactNode {
  const {pathname} = useLocation();
  const [edge, setEdge] = useState<number | null>(null);
  const dragging = useRef(false);

  // Where the handle sits: the sidebar's right edge, or nowhere when the page has no
  // docs sidebar or the window is too narrow for one.
  const place = useCallback((): void => {
    const sidebar = document.querySelector(SIDEBAR);
    if (!sidebar || window.innerWidth < DESKTOP) {
      setEdge(null);
      return;
    }
    setEdge(sidebar.getBoundingClientRect().right);
  }, []);

  useEffect(() => {
    place();
    const later = window.setTimeout(place, 300); // after the sidebar has laid out
    window.addEventListener('resize', place);
    return () => {
      window.clearTimeout(later);
      window.removeEventListener('resize', place);
    };
  }, [pathname, place]);

  const onPointerDown = (event: React.PointerEvent<HTMLDivElement>): void => {
    dragging.current = true;
    event.currentTarget.setPointerCapture(event.pointerId);
    document.documentElement.classList.add('tc-sidebar-dragging');
  };
  const onPointerMove = (event: React.PointerEvent<HTMLDivElement>): void => {
    const sidebar = document.querySelector(SIDEBAR);
    if (!dragging.current || !sidebar) {
      return;
    }
    setWidth(event.clientX - sidebar.getBoundingClientRect().left);
    place();
  };
  const onPointerUp = (event: React.PointerEvent<HTMLDivElement>): void => {
    dragging.current = false;
    event.currentTarget.releasePointerCapture(event.pointerId);
    document.documentElement.classList.remove('tc-sidebar-dragging');
  };
  const onDoubleClick = (): void => {
    const sidebar = document.querySelector(SIDEBAR);
    if (!sidebar) {
      return;
    }
    const fit = Math.min(fitWidth(sidebar), maxWidth());
    const current = Math.round(sidebar.getBoundingClientRect().width);
    setWidth(Math.abs(current - fit) <= 2 ? null : fit);
    window.setTimeout(place, 50);
    window.setTimeout(place, 400); // after the theme's width transition
  };

  if (edge === null) {
    return null;
  }
  return (
    <div
      className="tc-sidebar-resizer"
      style={{left: `${edge - 4}px`}}
      role="separator"
      aria-orientation="vertical"
      aria-label="Drag to resize the sidebar; double-click to fit it to its longest entry"
      title="Drag to resize. Double-click to fit the longest entry."
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={onDoubleClick}
    />
  );
}
