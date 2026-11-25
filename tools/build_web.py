"""Build a browser-only version of the app (no installation needed).

    python tools/build_web.py          # writes site/index.html

The page runs the same app.py and robot/ package with stlite
(Streamlit compiled to WebAssembly via Pyodide). Open site/index.html through
any static web server, or use the published GitHub Pages copy.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STLITE_VERSION = "1.12.1"  # pinned; bundles Streamlit 1.65 on Python 3.14
CDN = f"https://cdn.jsdelivr.net/npm/@stlite/browser@{STLITE_VERSION}/build"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Campus Delivery Robot</title>
<link rel="stylesheet" href="__CDN__/stlite.css">
<style>
  body { margin: 0; background: #ffffff; }
  #loading-note {
    position: fixed; left: 50%; bottom: 24px; transform: translateX(-50%);
    z-index: 1000; padding: 10px 18px; border-radius: 999px;
    background: #0f1b2d; color: #f6f4ee; font: 15px/1.4 system-ui, sans-serif;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  }
</style>
</head>
<body>
<div id="loading-note">Loading Python in your browser&hellip; first load takes ~20&ndash;40&nbsp;s</div>
<div id="root"></div>
<script id="app-files" type="application/json">__FILES__</script>
<script type="module">
import { mount } from "__CDN__/stlite.js";
const files = JSON.parse(document.getElementById("app-files").textContent);
mount(
  {
    entrypoint: "app.py",
    files,
    requirements: [],
    streamlitConfig: { "client.toolbarMode": "viewer" },
  },
  document.getElementById("root"),
);
// Hide the loading note as soon as the app has rendered its title.
const note = document.getElementById("loading-note");
const timer = setInterval(() => {
  if (document.body.innerText.includes("Campus Delivery Robot") &&
      document.querySelector('[data-testid="stAppViewContainer"]')) {
    note.remove();
    clearInterval(timer);
  }
}, 500);
</script>
</body>
</html>
"""


def collect_files() -> dict[str, str]:
    files = {"app.py": (ROOT / "app.py").read_text(encoding="utf-8")}
    for path in sorted((ROOT / "robot").glob("*.py")):
        files[f"robot/{path.name}"] = path.read_text(encoding="utf-8")
    return files


def build(out_dir: Path = ROOT / "site") -> Path:
    # "</" is escaped so no source file can close the <script> tag early.
    payload = json.dumps(collect_files(), ensure_ascii=False).replace("</", "<\\/")
    html = PAGE.replace("__CDN__", CDN).replace("__FILES__", payload)
    out_dir.mkdir(exist_ok=True)
    (out_dir / ".nojekyll").write_text("")
    target = out_dir / "index.html"
    target.write_text(html, encoding="utf-8")
    return target


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
