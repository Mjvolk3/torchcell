# experiments/tcdb-002-build-speed/scripts/profile_build2032.py
# [[experiments.tcdb-002-build-speed.scripts.profile_build2032]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/profile_build2032
"""Pair job 2032's per-adapter windows (slurm log timestamps) with W&B system metrics."""

import json
import re
from datetime import datetime, timedelta, timezone

import pandas as pd
import wandb

LOG = "/scratch/projects/torchcell/database/slurm/output/2032_live_rebuild_kg.out"
RUN = "zhao-group/tcdb/6etw9rur"
TZ = timezone(timedelta(hours=-5))  # GilaHyper local, log stamps are local

pat = re.compile(
    r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+\]\[__main__\]\[INFO\] - Writing (nodes|edges) for adapter: (\w+)"
)
events = []
last_ts = None
with open(LOG) as f:
    for line in f:
        m = pat.match(line)
        if m:
            ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=TZ)
            events.append((ts, m.group(2), m.group(3)))
        if "Finished iterating" in line:
            last_ts = line[:20]
print("finished line:", last_ts)

# windows: each event runs until the next event
rows = []
for i, (ts, kind, name) in enumerate(events):
    end = events[i + 1][0] if i + 1 < len(events) else None
    rows.append({"adapter": name, "phase": kind, "start": ts, "end": end})
win = pd.DataFrame(rows)
win["sec"] = (win["end"] - win["start"]).dt.total_seconds()

api = wandb.Api()
run = api.run(RUN)
sys_df = run.history(stream="events", samples=20000, pandas=True)
sys_df["t"] = pd.to_datetime(sys_df["_timestamp"], unit="s", utc=True)
cols = [c for c in sys_df.columns if c.startswith("system.")]
print("system columns:", cols)
cpu_col = "system.cpu" if "system.cpu" in sys_df else None
mem_cols = [c for c in cols if "memory" in c]
print("mem cols", mem_cols)


def stats(mask):
    d = sys_df[mask]
    out = {"n_samples": int(mask.sum())}
    for c in [
        "system.cpu",
        "system.proc.cpu.threads",
        "system.memory",
        "system.proc.memory.rssMB",
        "system.proc.memory.percent",
        "system.proc.memory.availableMB",
    ]:
        if c in d and d[c].notna().any():
            out[c + ".mean"] = round(float(d[c].mean()), 1)
            out[c + ".max"] = round(float(d[c].max()), 1)
    return out


per_adapter = []
for name, g in win.groupby("adapter", sort=False):
    start = g["start"].min()
    end = g["end"].max()
    if pd.isna(end):
        continue
    mask = (sys_df["t"] >= start) & (sys_df["t"] < end)
    s = stats(mask)
    s.update(
        {
            "adapter": name,
            "nodes_sec": float(g[g.phase == "nodes"]["sec"].sum()),
            "edges_sec": float(g[g.phase == "edges"]["sec"].sum()),
        }
    )
    s["total_sec"] = s["nodes_sec"] + s["edges_sec"]
    per_adapter.append(s)
pa = pd.DataFrame(per_adapter).sort_values("total_sec", ascending=False)
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 30)
print(pa.to_string(index=False))
print("sum hours", pa["total_sec"].sum() / 3600)
pa.to_csv(
    "experiments/tcdb-002-build-speed/results/build2032_per_adapter.csv", index=False
)

# summary keys that carry the per-dataset lengths
summ = {k: v for k, v in run.summary.items() if k.endswith("_len")}
json.dump(
    summ,
    open("experiments/tcdb-002-build-speed/results/build2032_lens.json", "w"),
    indent=1,
)
print({k: summ[k] for k in sorted(summ, key=lambda k: -summ[k])[:12]})
