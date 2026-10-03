"""설정 로드와 재현용 식별자(config_hash, code_version, data_snapshot_id)."""
from __future__ import annotations

import glob
import hashlib
import os
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:12]


def load_config(path: str | None = None) -> dict:
    path = path or os.path.join(ROOT, "config", "config.toml")
    with open(path, "rb") as f:
        raw = f.read()
    cfg = tomllib.loads(raw.decode("utf-8"))
    cfg["_config_hash"] = _sha(raw)
    cfg["_config_path"] = path
    return cfg


def code_version() -> str:
    """git 이 있으면 커밋 해시, 없으면 src/*.py 내용 해시."""
    head = os.path.join(ROOT, ".git", "HEAD")
    if os.path.exists(head):
        try:
            ref = open(head).read().strip()
            if ref.startswith("ref:"):
                ref = open(os.path.join(ROOT, ".git", ref.split()[1])).read().strip()
            return ref[:12]
        except OSError:
            pass
    h = hashlib.sha256()
    for f in sorted(glob.glob(os.path.join(ROOT, "src", "*.py"))) + sorted(glob.glob(os.path.join(ROOT, "*.py"))):
        h.update(open(f, "rb").read())
    return "src-" + h.hexdigest()[:12]


def data_snapshot_id(files: list[str]) -> str:
    h = hashlib.sha256()
    for f in sorted(files):
        h.update(os.path.basename(f).encode())
        h.update(open(f, "rb").read())
    return h.hexdigest()[:12]


def abspath(cfg: dict, key: str) -> str:
    p = cfg["paths"][key]
    return p if os.path.isabs(p) else os.path.join(ROOT, p)
