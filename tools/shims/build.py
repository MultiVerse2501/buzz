"""Build the ARM64 shim wheels in tools/shims/ into the tools/wheels/ index.

Each shim project is built with ``uv build --wheel`` (produces a universal
``py3-none-any`` pure-Python wheel), then re-tagged to
``py3-none-win_arm64`` so it can only ever satisfy resolution on Windows
ARM64 — the platform where the real torchaudio/sphn have no wheels. The
retag rewrites the ``WHEEL`` tag line and regenerates ``RECORD`` hashes so
the distribution stays fully verifiable. The finished wheels land in
``tools/wheels/``, the find-links index pyproject.toml references under
the ARM64 marker.

Usage: ``python tools/shims/build.py``
"""

import base64
import hashlib
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "wheels"
SHIMS = ("sphn", "torchaudio")


def retag(wheel_path: Path, tag_from: str, tag_to: str) -> Path:
    target = wheel_path.with_name(wheel_path.name.replace(tag_from, tag_to))
    with zipfile.ZipFile(wheel_path) as zin:
        entries = []
        dist_info = next(
            n.split("/")[0] + "/" for n in zin.namelist() if n.split("/")[0].endswith(".dist-info")
        )
        for name in zin.namelist():
            data = zin.read(name)
            if name == f"{dist_info}WHEEL":
                data = data.replace(f"Tag: {tag_from}".encode(), f"Tag: {tag_to}".encode())
            if not name.endswith(f"{dist_info}RECORD"):
                entries.append((name, data))
        record_name = f"{dist_info}RECORD"
        record_lines = [
            f"{name},sha256="
            + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            + f",{len(data)}"
            for name, data in entries
        ]
        entries.append((record_name, ("".join(line + "\n" for line in record_lines)).encode()))
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as zout:
            for name, data in entries:
                zout.writestr(name, data)
    return target


def main() -> int:
    OUT.mkdir(exist_ok=True)
    for shim in SHIMS:
        shim_dir = ROOT / shim
        dist = shim_dir / "dist"
        if dist.exists():
            shutil.rmtree(dist)
        subprocess.run(["uv", "build", "--wheel"], cwd=shim_dir, check=True)
        universal = next(dist.glob("*-py3-none-any.whl"))
        tagged = retag(universal, "py3-none-any", "py3-none-win_arm64")
        shutil.move(str(tagged), OUT / tagged.name)
        shutil.rmtree(dist)
        print(f"built {OUT / tagged.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
