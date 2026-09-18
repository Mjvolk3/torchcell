# database/browser/patch_browser_jar.py
# [[database.browser.patch_browser_jar]]
# https://github.com/Mjvolk3/torchcell/tree/main/database/browser/patch_browser_jar
"""Put the torchcell styling seed into the Neo4j Browser jar.

Neo4j serves the Browser from ``lib/neo4j-browser-<version>.jar`` (static files under
``browser/``), and its page sends a Content-Security-Policy that allows scripts from the
server itself and nothing inline. So the seed is added to the jar as
``browser/torchcell-seed.js`` and ``browser/index.html`` gains one ``<script>`` tag for it
ahead of the Browser's own module script. Run inside the image build by
``database/docker/Dockerfile.tc-neo4j-browser``:

    python patch_browser_jar.py /var/lib/neo4j/lib torchcell-seed.js

Idempotent: a second run replaces the seed and leaves the already-tagged page alone.
Stdlib only, since it runs with the image's python.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

INDEX_ENTRY = "browser/index.html"
SEED_ENTRY = "browser/torchcell-seed.js"
SEED_TAG = '    <script src="./torchcell-seed.js"></script>\n'
ANCHOR = '    <script type="module" crossorigin src="./assets/index-'


def browser_jar(lib_dir: Path) -> Path:
    """The one Browser jar in the Neo4j lib directory."""
    jars = sorted(lib_dir.glob("neo4j-browser-*.jar"))
    if len(jars) != 1:
        raise FileNotFoundError(
            f"expected one neo4j-browser-*.jar in {lib_dir}, found {jars}"
        )
    return jars[0]


def patched_index(html: str) -> str:
    """``index.html`` with the seed script tag ahead of the Browser's module script."""
    if SEED_TAG in html:
        return html
    if html.count(ANCHOR) != 1:
        raise ValueError(f"{INDEX_ENTRY} has no single line starting {ANCHOR!r}")
    return html.replace(ANCHOR, SEED_TAG + ANCHOR, 1)


def patch_jar(jar: Path, seed: Path) -> None:
    """Rewrite ``jar`` in place with the seed added and the page tagged."""
    seed_text = seed.read_text(encoding="utf-8")
    with tempfile.NamedTemporaryFile(
        dir=jar.parent, suffix=".jar", delete=False
    ) as tmp:
        tmp_path = Path(tmp.name)
    with zipfile.ZipFile(jar) as src, zipfile.ZipFile(tmp_path, "w") as dst:
        names = src.namelist()
        if INDEX_ENTRY not in names:
            raise FileNotFoundError(f"{jar} has no {INDEX_ENTRY}")
        for info in src.infolist():
            if info.filename == SEED_ENTRY:
                continue
            data = src.read(info)
            if info.filename == INDEX_ENTRY:
                data = patched_index(data.decode("utf-8")).encode("utf-8")
            dst.writestr(info, data, compress_type=info.compress_type)
        dst.writestr(SEED_ENTRY, seed_text, compress_type=zipfile.ZIP_DEFLATED)
    shutil.copymode(jar, tmp_path)
    tmp_path.replace(jar)


def main(argv: list[str] | None = None) -> int:
    """Patch the Browser jar under a Neo4j lib directory with a seed script."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "lib_dir", type=Path, help="Neo4j lib directory holding the Browser jar"
    )
    parser.add_argument("seed", type=Path, help="torchcell-seed.js to add")
    args = parser.parse_args(argv)
    jar = browser_jar(args.lib_dir)
    patch_jar(jar, args.seed)
    with zipfile.ZipFile(jar) as check:
        html = check.read(INDEX_ENTRY).decode("utf-8")
        assert SEED_TAG in html and SEED_ENTRY in check.namelist()
    print(f"patched {jar}: {SEED_ENTRY} added, {INDEX_ENTRY} tagged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
