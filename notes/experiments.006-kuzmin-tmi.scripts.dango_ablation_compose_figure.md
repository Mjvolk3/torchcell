---
id: b18581ze30dpewcsk7lgff9
title: Dango_ablation_compose_figure
desc: ''
updated: 1793156403919
created: 1793156403919
---

## 2026.10.27 - Composing FigS-dango-ablation

`experiments/006-kuzmin-tmi/scripts/dango_ablation_compose_figure.py` writes `notes/assets/drawio/FigS-dango-ablation.drawio` from the three third-width panel SVGs of [[experiments.006-kuzmin-tmi.scripts.dango_ablation_si]]: one row, (a) validation curves, (b) best validation and test Pearson r per arm, (c) validation reconstruction loss. It refuses any panel that is not third width (57.8 mm = 227.56 units, tolerance 0.5 unit). White-cross constants copied from [[experiments.010-kuzmin-tmi.scripts.dango_full_dataset_compose_figure]]: `COL_GAP = 12`, `TOP_STRIP = 16`, letters fontSize 11.1 bold at `(panel_x, row_top)`.

`--export` runs the headless draw.io export under a private D-Bus session (`env -u DBUS_SESSION_BUS_ADDRESS timeout 180 dbus-run-session -- xvfb-run -a $DRAWIO FILE.drawio --password-store=basic --no-sandbox --disable-gpu --disable-dev-shm-usage -x -f pdf --crop -o paper/nature-biotech/figures/FigS-dango-ablation.pdf`, with `DRAWIO=/home/michaelvolk/.local/opt/drawio-deb/opt/drawio/drawio` on GilaHyper), requires the PDF to be newer than a stamp taken before the call, and then runs `paper/nature-biotech/scripts/drawio_font_band.py --check`.

Figure 707 x 221 units = 179.5 x 56.1 mm; exported PDF 510 x 160.08 pt = 179.9 x 56.5 mm (export 28.8 s). `drawio_font_band.py --check` reports the 3 panel letters at 7.99 pt and nothing else; the thesis `common/drawio_text_clearance.py --check` reports all 3 labels clear every line by 0.4 mm; `check-figures.sh` passes. Fonts in the export: ArialMT, Arial-BoldMT, and the LiberationSerif space-glyph subset draw.io writes for an embedded image. In `editing-nature-biotech.pdf` it is Supplementary Fig. 10, on one page with its caption and Supplementary Table 10 (page 48 of the 2026.10.27 build); the review page is `paper/nature-biotech/review/FigS-dango-ablation.pdf` (ignored by git).

Piping the export through `grep` in zsh hung the shell for 6 min after the export had finished; the export itself does not hang. Each export also leaves a `gnome-keyring-daemon` behind (D-Bus activation), and hundreds of these from earlier exports were running on 2026-10-27.
