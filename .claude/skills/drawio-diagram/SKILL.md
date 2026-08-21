---
name: drawio-diagram
description: Author a NEW draw.io diagram from scratch by writing the .drawio XML directly -- at true Nature print size, in the repo palette, verified by headless render. Use when asked to draw/diagram a mechanism, workflow, schematic, or concept figure that is not a matplotlib plot. For refreshing plot panels already placed in a figure, use fig-swap instead.
---

A `.drawio` file is plain mxGraph XML, so a diagram can be authored programmatically --
no GUI. This skill is for **hand-composed schematics** (mechanisms, workflows, state
diagrams). Its sibling `fig-swap` maintains matplotlib panels already embedded in a
figure; do not confuse them.

The whole job is: pick the size first, write the XML, then **render and look at it**.
Step 4 is not optional -- text wrapping cannot be predicted, only measured.

## Step 0: Fix the size before writing anything

**100 draw.io units = 1 inch**, so `units = mm x 3.9392`. This is verifiable, not
folklore: the "179.4 mm" reference box in
`notes/assets/drawio/figure-sizing-template.drawio.svg` measures 706.69 units.

| Slot | mm | units |
| --- | --- | --- |
| full | 179.4 | 706.7 |
| wide | 118.9 | 468.3 |
| half_plus | 88.5 | 348.4 |
| half | 88 | 346.5 |
| third | 57.8 | 227.6 |
| max height | 170 | 669.6 |

Constants live in `torchcell/utils/utils.py` (`PANEL_WIDTHS_MM`, `MAX_HEIGHT_MM`).
**Design to the cap minus ~4 units** -- draw.io adds a 1-unit border per side on export,
so content at 706 exports at 708 and busts the cap. 702 is a safe full-width target.

Reference to match: `notes/assets/drawio/Fig1-torchcell-overview.drawio.svg` is exactly
707 x 446 units.

## Step 1: Type and colour

- **Arial everywhere.** Put `fontFamily=Arial;` in every cell's style string.
- **Font size is in units, so 6 pt (the Nature minimum) = 6/72 in = 8.33 units.** Use
  `fontSize=8.5` for body text and labels, `fontSize=12` (8.6 pt) for a title. Never go
  below 8.5, and never shrink the font to fix an overflow -- cut words instead.
- **Palette: slots 1-6 of `PLOT_PALETTE` / `PLOT_PALETTE_FILL` only** (`strokeColor` /
  `fillColor`). Never invent colours; keep it green-free.

  | | stroke | fill |
  | --- | --- | --- |
  | orange | `#D79B00` | `#FFE6CC` |
  | red | `#B85450` | `#F8CECC` |
  | purple | `#9673A6` | `#E1D5E7` |
  | yellow | `#D6B656` | `#FFF2CC` |
  | blue | `#6C8EBF` | `#DAE8FC` |
  | gray | `#666666` | `#F5F5F5` |

- Spend the four warm primaries before blue/gray. Past 6 categories, disambiguate with
  dashes or `strokeWidth`, not new colours.

## Step 2: Write the XML

```xml
<mxfile host="app.diagrams.net" agent="claude-code">
  <diagram name="NAME" id="NAME">
    <mxGraphModel dx="1400" dy="1000" grid="0" page="1" pageWidth="703" pageHeight="639" math="0" shadow="0">
      <root>
        <mxCell id="0" /><mxCell id="1" parent="0" />
        <mxCell id="box" value="label" style="rounded=0;whiteSpace=wrap;html=1;fillColor=#FFE6CC;strokeColor=#D79B00;fontFamily=Arial;fontSize=8.5;" vertex="1" parent="1">
          <mxGeometry x="0" y="0" width="150" height="22" as="geometry" />
        </mxCell>
        <mxCell id="arr" style="endArrow=classic;html=1;strokeWidth=1.5;strokeColor=#666666;endSize=4;" edge="1" parent="1">
          <mxGeometry relative="1" as="geometry">
            <mxPoint x="75" y="26" as="sourcePoint" />
            <mxPoint x="75" y="60" as="targetPoint" />
          </mxGeometry>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>
```

Edges take explicit `sourcePoint`/`targetPoint`; they need not attach to shapes.

**Escaping -- the usual reason a hand-written file renders as literal tag text: `value`
is an XML attribute, so markup inside it must be escaped** (`&lt;b&gt;bold&lt;/b&gt;`,
`&lt;i&gt;`). Use entities for symbols: `&#8242;` prime, `&#176;` degree, `&#181;` micro,
`&#946;` beta, `&#8776;` approx, `&#215;` times, `&#10;` newline-inside-a-label. **No
em-dashes** (repo rule) -- write ` -- `.

**Define a column grid once, in a comment at the top of the file, and reuse it** across
every row. Rows that share x-columns align; eyeballed x-values do not.

## Step 3: Export headless

```bash
DRAWIO=/Applications/draw.io.app/Contents/MacOS/draw.io   # Linux: xvfb-run <AppImage> --no-sandbox
"$DRAWIO" --no-sandbox -x -f svg -o /tmp/check.svg  FIG.drawio    # to MEASURE
"$DRAWIO" --no-sandbox -x -f png -s 4 -o /tmp/check.png FIG.drawio  # to LOOK
```

Verify the cap mechanically and read the numbers -- do not assume:

```bash
python3 -c "
import re; s=open('/tmp/check.svg').read()
w,h=[float(x) for x in re.search(r'width=\"([\d.]+)px\" height=\"([\d.]+)px\"',s).groups()]
U=706.6915/179.4
print(f'{w/U:.2f} x {h/U:.2f} mm', 'OK' if w/U<=179.4 and h/U<=170 else 'OVER')"
```

## Step 4: Look at the PNG. Always. Then fix and repeat.

Read the exported PNG with the Read tool and actually inspect it. **Rendered text height
cannot be predicted from character counts** -- expect the first pass to have annotation
blocks overrunning into the row below.

- Budget **~11 units per rendered line** at `fontSize=8.5`.
- Count the real lines in the render, set each text cell's height from that, then
  re-lay-out every row beneath it. Two or three iterations is normal.
- If the re-layout busts the height cap, **cut prose**. In a paper figure, long
  explanation belongs in the caption anyway.

Report done only after a render you have looked at.

## Step 5: Wire it in

- Save as `notes/assets/drawio/<subject-name>.drawio`. **Name it for its subject**, not
  for the tool or the session (`splitseq-barcoding.drawio`, not `diagram-test.drawio`).
- `notes-tex/<doc>/`: reference `figures/<name>.pdf`; `make figures` finds
  `$REPO/notes/assets/drawio/<name>.{drawio.svg,drawio.png,drawio}` by matching basename
  and exports a cropped PDF. Then `make && make check` -- the gate prints the figure
  width in mm and fails on any `\cite` key missing from `references.bib`.
- `paper/nature-biotech/`: export into `paper/nature-biotech/figures/`, then
  `make -C paper/nature-biotech paper` (auto-re-exports draw.io figs on macOS).
- **The caption must carry provenance**: `\citep{key}` for the source the content came
  from, a `\file{...}` pointer to the `.drawio`, and -- when you composed the diagram
  from prose rather than redrawing a published figure -- an explicit statement of that.
  Add the key to the Zotero collection and run `make bib` BEFORE citing, or `make check`
  fails.

## Hazards

- **Never write ` -- ` inside an XML comment.** A double hyphen anywhere in
  `<!-- ... -->` is not well-formed XML, and draw.io fails the export with only
  `Error: Export failed`, naming no line. This repo's house style uses ` -- ` for
  dashes, so header comments walk straight into it -- use a colon or semicolon there
  instead. Validate before exporting; it names the line and the export does not:
  `python3 -c "import xml.dom.minidom as m; m.parse('FIG.drawio'); print('XML OK')"`
- **Put `whiteSpace=wrap;` in EVERY text cell's style.** A `text;html=1;` cell without it
  does not wrap, and its rendered label -- not its `mxGeometry` -- sets the export
  bounds. One 38-character legend label at x=596 silently widened a 702-unit figure to
  752. The symptom is an export wider than any shape in the file, so when the measured
  width exceeds every rect's right edge, look for an unwrapped label, not a stray shape.
- **The file must be CLOSED in draw.io / the VS Code draw.io editor while you edit it.**
  draw.io saves whole-file from memory and will silently revert scripted edits.
- The GUI reformats the XML on save (indentation, attribute order, `host=`). That is
  harmless -- treat the on-disk version as current, do not revert it.
- `-s 4` PNG is for inspection only; the SVG/PDF export is the artifact.
- Do not hand-author something that is really a plot. If it contains data, it belongs in
  a script under `experiments/<id>/` (repo rule), rendered with matplotlib and placed
  with `fig-swap`.
