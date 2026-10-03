"""12절 로그: 추가만 하고 덮어쓰지 않는다. 실행마다 run_id 파일 하나."""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone

import pandas as pd

from .config import ROOT


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def load_schema(table: str) -> dict:
    with open(os.path.join(ROOT, "schema", f"{table}.json"), encoding="utf-8") as f:
        return json.load(f)


def write_log(rows: list[dict], table: str, log_dir: str, run_id: str) -> str:
    schema = load_schema(table)
    cols = list(schema["columns"].keys())
    df = pd.DataFrame(rows)
    missing = [c for c in cols if c not in df.columns]
    extra = [c for c in df.columns if c not in cols]
    if missing or extra:
        raise ValueError(f"{table}: schema mismatch missing={missing} extra={extra}")
    df = df[cols]
    d = os.path.join(log_dir, table)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"run_{run_id}.parquet")
    if os.path.exists(path):
        raise FileExistsError(path)  # append-only
    df.to_parquet(path, index=False)
    return path
