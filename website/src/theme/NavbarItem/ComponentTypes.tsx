import ComponentTypes from '@theme-original/NavbarItem/ComponentTypes';
import WidthToggle from '@site/src/components/WidthToggle';

// Adds the page-width toggle as a navbar item type. A custom type must be named
// `custom-...`; docusaurus.config.ts places it with `{type: 'custom-widthToggle'}`.
export default {
  ...ComponentTypes,
  'custom-widthToggle': WidthToggle,
};
