"""The Browser jar patch adds the seed, tags the page once, and is idempotent."""

from __future__ import annotations

import importlib.util
import zipfile
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "database" / "browser" / "patch_browser_jar.py"
INDEX_HTML = (
    "<!doctype html>\n<html>\n  <head>\n    <title>Neo4j Browser</title>\n"
    '    <script type="module" crossorigin src="./assets/index-Dez4Av7l.js"></script>\n'
    '    <link rel="modulepreload" crossorigin href="./assets/vendor-react.js">\n'
    "  </head>\n  <body></body>\n</html>\n"
)


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("patch_browser_jar", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_jar(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as jar:
        jar.writestr("META-INF/MANIFEST.MF", "Manifest-Version: 1.0\n")
        jar.writestr("browser/index.html", INDEX_HTML)
        jar.writestr("browser/assets/index-Dez4Av7l.js", "console.log(1)\n")


def test_patch_adds_seed_and_tags_index_once(tmp_path: Path) -> None:
    mod = _load()
    lib = tmp_path / "lib"
    lib.mkdir()
    jar = lib / "neo4j-browser-2026.06.30+0.jar"
    _synthetic_jar(jar)
    seed = tmp_path / "torchcell-seed.js"
    seed.write_text("window.localStorage.setItem('k', 'v');\n", encoding="utf-8")

    assert mod.main([str(lib), str(seed)]) == 0
    with zipfile.ZipFile(jar) as patched:
        names = patched.namelist()
        assert "browser/torchcell-seed.js" in names
        assert "browser/assets/index-Dez4Av7l.js" in names
        html = patched.read("browser/index.html").decode("utf-8")
        assert (
            patched.read("browser/torchcell-seed.js").decode("utf-8")
            == seed.read_text()
        )
    assert html.count("torchcell-seed.js") == 1
    assert html.index("torchcell-seed.js") < html.index("./assets/index-Dez4Av7l.js")

    # second run: page unchanged, seed replaced with the new content
    seed.write_text("window.localStorage.setItem('k', 'v2');\n", encoding="utf-8")
    assert mod.main([str(lib), str(seed)]) == 0
    with zipfile.ZipFile(jar) as patched:
        assert patched.namelist().count("browser/torchcell-seed.js") == 1
        assert patched.read("browser/index.html").decode("utf-8") == html
        assert "v2" in patched.read("browser/torchcell-seed.js").decode("utf-8")


def test_patch_refuses_an_unexpected_page(tmp_path: Path) -> None:
    mod = _load()
    lib = tmp_path / "lib"
    lib.mkdir()
    jar = lib / "neo4j-browser-9.jar"
    with zipfile.ZipFile(jar, "w") as raw:
        raw.writestr("browser/index.html", "<html><body>no module script</body></html>")
    seed = tmp_path / "seed.js"
    seed.write_text("1;\n", encoding="utf-8")
    try:
        mod.patch_jar(jar, seed)
    except ValueError as exc:
        assert "index.html" in str(exc)
    else:
        raise AssertionError("a page without the module script line must be refused")
