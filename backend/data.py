"""Dataset download / caching.

The original Kaggle notebook reads from
``/kaggle/input/.../martj42/international-football-results-from-1872-to-2017``.
That exact dataset is mirrored (and kept up to date) in the author's public
GitHub repo, so the webapp pulls the CSVs from there and caches them locally.
This makes the app self-contained: no Kaggle account or manual download needed.
"""

from __future__ import annotations

import os
import time
import urllib.request

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")

# martj42/international_results — the same source Kaggle re-publishes.
BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"

FILES = {
    "results.csv": f"{BASE_URL}/results.csv",
    "shootouts.csv": f"{BASE_URL}/shootouts.csv",
    "goalscorers.csv": f"{BASE_URL}/goalscorers.csv",
}


# The egress proxy truncates large streamed responses (~2.8 MB), so files are
# fetched in smaller byte ranges and stitched back together. GitHub raw serves
# HTTP 206 / Range requests, which makes this reliable.
CHUNK = 1_000_000  # 1 MB per range request — comfortably under the proxy limit


def _content_length(url: str) -> int | None:
    req = urllib.request.Request(
        url, method="HEAD", headers={"User-Agent": "wc2026predictor"}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None


def _get_range(url: str, start: int, end: int) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "wc2026predictor", "Range": f"bytes={start}-{end}"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def _download(url: str, dest: str, retries: int = 4) -> None:
    """Download a URL to ``dest`` in byte-range chunks, with retries.

    Works around proxy truncation of large responses by fetching ``CHUNK``-sized
    ranges and concatenating them, then verifying the total against the
    server-reported Content-Length.
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            total = _content_length(url)
            buf = bytearray()
            start = 0
            while True:
                end = (start + CHUNK - 1) if total is None else min(start + CHUNK - 1, total - 1)
                part = _get_range(url, start, end)
                if not part:
                    break
                buf.extend(part)
                start += len(part)
                if total is not None and start >= total:
                    break
                if total is None and len(part) < CHUNK:
                    break
            if total is not None and len(buf) < total:
                raise IOError(f"short read: {len(buf)}/{total} bytes")
            with open(dest, "wb") as f:
                f.write(buf)
            return
        except Exception as err:  # noqa: BLE001 - retry any transient failure
            last_err = err
            wait = 2 ** attempt
            print(f"[data] attempt {attempt + 1} failed ({err}); retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to download {url}: {last_err}")


def ensure_data(force: bool = False) -> dict[str, str]:
    """Download the CSVs into ``data/raw`` if not already present.

    Returns a mapping of logical name -> local file path.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    paths: dict[str, str] = {}
    for name, url in FILES.items():
        dest = os.path.join(RAW_DIR, name)
        if force or not os.path.exists(dest) or os.path.getsize(dest) == 0:
            print(f"[data] downloading {name} ...")
            _download(url, dest)
        paths[name] = dest
    return paths


if __name__ == "__main__":
    for name, path in ensure_data(force=True).items():
        size = os.path.getsize(path)
        print(f"  {name}: {size:,} bytes -> {path}")
