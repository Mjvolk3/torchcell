# env/tested

`requirements.lock.txt` is the full resolution (every installed package as `name==version`, sorted, torchcell itself and the job's pip and uv left out) of the last weekly dependency run whose `pytest tests/torchcell` passed: the last known good set of versions on Python 3.13.0 with the CPU torch build in `env/torch-ci-version.txt`.
It is written only by `scripts/deps_weekly.py` (the weekly `run`, or `write-lock` from a run's report), never by hand, and each weekly run diffs its new resolution against it to report which packages moved.
To reproduce that environment, install torch from the CPU index first, then `pip install -r env/tested/requirements.lock.txt -f https://data.pyg.org/whl/torch-<version>+cpu.html`.
