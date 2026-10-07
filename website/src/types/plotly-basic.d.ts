// plotly.js-basic-dist-min ships a prebuilt bundle with no type declarations. It
// exposes the same API as plotly.js, restricted to the scatter, bar, and pie traces.
declare module 'plotly.js-basic-dist-min' {
  import * as Plotly from 'plotly.js';
  export default Plotly;
}
