"""Wait for the exact runtime wheel on PyPI and verify it matches this release."""

import argparse
import io
import json
from pathlib import Path
import time
import tomllib
import urllib.error
import urllib.request
import zipfile


def payload(data: bytes) -> dict[str, bytes]:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        # RECORD contains archive hashes; compare the actual contents instead.
        return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/RECORD")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist-dir", type=Path, default=Path("dist"))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    version = tomllib.loads((root / "packages/training-runtime/pyproject.toml").read_text())["project"]["version"]
    filename = f"smithtune_training_runtime-{version}-py3-none-any.whl"
    expected = payload((args.dist_dir / filename).read_bytes())
    url = f"https://pypi.org/pypi/smithtune-training-runtime/{version}/json"
    for attempt in range(36):
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                metadata = json.load(response)
            wheel = next((item for item in metadata["urls"] if item["filename"] == filename), None)
            if wheel is not None:
                with urllib.request.urlopen(wheel["url"], timeout=60) as response:
                    actual = payload(response.read())
                if expected != actual:
                    raise SystemExit("Published runtime differs from this build; bump its version before releasing smithtune.")
                print(f"Verified published runtime {version} matches the tested wheel.")
                return
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise
        print(f"Waiting for runtime {version} on PyPI ({attempt + 1}/36)", flush=True)
        time.sleep(5)
    raise SystemExit("Runtime is not available on PyPI; smithtune was not published.")


if __name__ == "__main__":
    main()
