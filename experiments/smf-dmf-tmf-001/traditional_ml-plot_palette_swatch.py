# experiments/smf-dmf-tmf-001/traditional_ml-plot_palette_swatch.py
# Emits an SVG (+ PNG preview) of the 17-encoding plot palette: base color (validation)
# next to its lighter version (test = base at 45% opacity over white), in canonical size
# order. Lets us judge whether any test-tint overlaps another encoding's base color.
import os, os.path as osp
from dotenv import load_dotenv
load_dotenv()
ASSET_IMAGES_DIR = os.getenv("ASSET_IMAGES_DIR")

COLORS = ['#BD8800','#5F7DA8','#A24A46','#846592','#C7A24C','#46557E','#9E4C86','#6E5479',
          '#C08552','#6E93B8','#8A3B3B','#9C6EA0','#B5943F','#4F688B','#A85878','#5A5A5A','#7C5A86']
NAMES = ['random_1','random_10','codon_frequency','random_100','normalized_chrom_pathways',
 'calm','fudt_upstream','fudt_downstream','random_1000','prot_T5_all','prot_T5_no_dubious',
 'esm2_650M_all','esm2_650M_no_dubious','nt_window_5979','nt_window_three_prime_300',
 'nt_window_five_prime_1003','one_hot_gene']

def hex2rgb(h): return tuple(int(h[i:i+2],16) for i in (1,3,5))
def rgb2hex(t): return "#%02X%02X%02X" % tuple(int(round(c)) for c in t)
def tint(h, a=0.45):  # base at opacity a over white = test-bar appearance
    r,g,b = hex2rgb(h); return rgb2hex((r*a+255*(1-a), g*a+255*(1-a), b*a+255*(1-a)))
TESTS = [tint(c) for c in COLORS]

# overlap check: nearest OTHER base to each test-tint (small = potential confusion)
def dist(a,b):
    ra,ga,ba=hex2rgb(a); rb,gb,bb=hex2rgb(b); return ((ra-rb)**2+(ga-gb)**2+(ba-bb)**2)**.5
print("nearest-other-base to each test tint (dist; <40 = watch):")
for i,(n,t) in enumerate(zip(NAMES,TESTS)):
    d,j = min((dist(t,COLORS[k]),k) for k in range(len(COLORS)) if k!=i)
    flag = "  <-- close to %s" % NAMES[j] if d<40 else ""
    print(f"  {n:26s} tint {t}  d={d:5.1f}{flag}")

rh, top = 34, 70
W, H = 660, top + len(COLORS)*rh + 20
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="Arial, sans-serif">',
       f'<rect width="{W}" height="{H}" fill="white"/>',
       f'<text x="20" y="30" font-size="17" font-weight="bold" fill="#222">Traditional-ML plot palette (17 encodings, green-free)</text>',
       f'<text x="300" y="56" font-size="12" font-weight="bold" fill="#444">val (base)</text>',
       f'<text x="470" y="56" font-size="12" font-weight="bold" fill="#444">test (lighter)</text>']
for i,(n,c,t) in enumerate(zip(NAMES,COLORS,TESTS)):
    y = top + i*rh
    svg.append(f'<text x="20" y="{y+18}" font-size="12" fill="#333">{n}</text>')
    svg.append(f'<rect x="290" y="{y}" width="150" height="26" rx="3" fill="{c}" stroke="#333" stroke-width="0.8"/>')
    svg.append(f'<text x="365" y="{y+17}" font-size="9.5" text-anchor="middle" fill="#fff">{c}</text>')
    svg.append(f'<rect x="460" y="{y}" width="150" height="26" rx="3" fill="{t}" stroke="#333" stroke-width="0.8"/>')
    svg.append(f'<text x="535" y="{y+17}" font-size="9.5" text-anchor="middle" fill="#333">{t}</text>')
svg.append('</svg>')
out_svg = osp.join(ASSET_IMAGES_DIR, "tradml-plot-palette.svg")
open(out_svg,"w").write("\n".join(svg)); print("\nwrote", out_svg)

# PNG preview
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(7.2, 6.6))
for i,(n,c,t) in enumerate(zip(NAMES,COLORS,TESTS)):
    yy = len(COLORS)-1-i
    ax.add_patch(plt.Rectangle((0,yy),1,0.86,color=c)); ax.add_patch(plt.Rectangle((1.05,yy),1,0.86,color=t))
    ax.text(-0.1, yy+0.43, n, ha="right", va="center", fontsize=9)
ax.text(0.5, len(COLORS)+0.1, "val (base)", ha="center", fontsize=10, fontweight="bold")
ax.text(1.55, len(COLORS)+0.1, "test (lighter)", ha="center", fontsize=10, fontweight="bold")
ax.set_xlim(-2.2, 2.1); ax.set_ylim(0, len(COLORS)+0.6); ax.axis("off")
png = "/private/tmp/claude-501/-Users-michaelvolk-Documents-projects-torchcell/3bb26517-92a1-434a-82fc-2acdc4990cc0/scratchpad/palette_swatch.png"
fig.savefig(png, dpi=140, bbox_inches="tight"); print("preview:", png)
