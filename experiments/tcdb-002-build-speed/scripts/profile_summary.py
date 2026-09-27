# experiments/tcdb-002-build-speed/scripts/profile_summary.py
# [[experiments.tcdb-002-build-speed.scripts.profile_summary]]
# https://github.com/Mjvolk3/torchcell/tree/main/experiments/tcdb-002-build-speed/scripts/profile_summary
"""Summarize a PROFILE=1 arm's py-spy speedscope file: main thread self and inclusive time.

    python experiments/tcdb-002-build-speed/scripts/profile_summary.py <run dir>

Writes ``results/<job>_profile_top.txt`` and prints it. The main thread is the profile
with the largest total weight (py-spy records every thread; the wandb and sampler
threads are mostly idle).
"""

import collections
import json
import os.path as osp
import sys
from pathlib import Path

RESULTS = Path(osp.dirname(osp.dirname(osp.abspath(__file__)))) / "results"
INTERESTING = (
    "biocypher",
    "torchcell",
    "concurrent",
    "multiprocessing",
    "csv",
    "pickle",
)


def main(run_dir: str) -> None:
    """Print the top self-time and inclusive frames of the main thread."""
    profile = json.load(open(osp.join(run_dir, "parent.speedscope")))
    frames = profile["shared"]["frames"]
    main_thread = max(profile["profiles"], key=lambda p: sum(p["weights"]))
    total = sum(main_thread["weights"])
    self_t: collections.Counter[int] = collections.Counter()
    incl: collections.Counter[int] = collections.Counter()
    for stack, weight in zip(
        main_thread["samples"], main_thread["weights"], strict=True
    ):
        if not stack:
            continue
        self_t[stack[-1]] += weight
        for frame in set(stack):
            incl[frame] += weight

    def name(i: int) -> str:
        f = frames[i]
        return f"{f.get('name')}  ({str(f.get('file', '')).split('/')[-1]}:{f.get('line', '')})"

    lines = [
        f"main thread {main_thread.get('name')}: {total:.0f} s sampled (py-spy --idle, 50 Hz)",
        "",
        "top self time",
    ]
    lines += [f"{w / total * 100:5.1f}%  {name(i)}" for i, w in self_t.most_common(25)]
    lines += ["", "top inclusive, biocypher / torchcell / executor frames"]
    shown = 0
    for i, w in incl.most_common(200):
        if any(k in str(frames[i].get("file", "")) for k in INTERESTING):
            lines.append(f"{w / total * 100:5.1f}%  {name(i)}")
            shown += 1
        if shown >= 35:
            break
    text = "\n".join(lines) + "\n"
    job = osp.basename(run_dir.rstrip("/")).split("_")[0]
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{job}_profile_top.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
