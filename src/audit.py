"""12절 감사 중 검정 단계에 해당하는 것.

1. 시간 인과성: 모든 행에서 window_end <= T0 (분봉 마감 기준).
2. 재현: 무작위로 고른 윈도우 하나를 같은 시드로 다시 계산해 저장값과 비교.
3. 스키마·키 유일성: (trading_date, candidate, version) 중복 없음.
수신 지연(receive_ts > T0)은 경고로만 보고한다 (마지막 봉이 T0 직후 수 초 뒤 도착).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_log(df: pd.DataFrame) -> dict:
    res = {}
    t0 = pd.to_datetime(df["t0_utc"])
    we = pd.to_datetime(df["window_end_utc"])
    res["causality_ok"] = bool((we <= t0).all())
    res["causality_violations"] = int((we > t0).sum())
    keys = ["trading_date", "candidate", "version"] + (["gate_rule"] if "gate_rule" in df else [])
    res["unique_key_ok"] = not df.duplicated(keys).any()
    late = df["late_receive_bars"].fillna(0)
    res["late_receive_rows"] = int((late > 0).sum())
    res["late_receive_max_sec"] = float(df["late_receive_max_sec"].fillna(0).max())
    res["warnings"] = []
    if res["late_receive_rows"]:
        res["warnings"].append(
            f"{res['late_receive_rows']} rows: 윈도우 마지막 분봉 수신 시각이 T0 이후 (최대 {res['late_receive_max_sec']:.1f}초). "
            "수신 기준을 엄격히 적용하면 07:59 봉은 T0 에 아직 없다.")
    return res


def reproduce_check(cfg, px, dates, run_fn, df: pd.DataFrame, n=1, seed=0) -> dict:
    """저장된 행 중 계산된 것 n 개를 다시 계산해 adf_stat·p 를 비교."""
    comp = df[df["stats_computed"] & df["adf_stat"].notna()]
    if comp.empty:
        return {"reproduce_ok": None, "note": "계산된 행이 없어 재현 점검 생략"}
    rng = np.random.default_rng(seed)
    picks = comp.sample(n=min(n, len(comp)), random_state=int(rng.integers(1e9)))
    out = []
    for _, r in picks.iterrows():
        d = pd.Timestamp(r["trading_date"]).date()
        rows = run_fn(cfg, px, [d])
        m = [x for x in rows if x["candidate"] == r["candidate"] and x["version"] == r["version"]
             and x.get("gate_rule") == r.get("gate_rule")][0]
        same = all(np.isclose(m[k], r[k], equal_nan=True) for k in ("adf_stat", "adf_p", "kss_stat", "kss_p", "b", "minp_p"))
        out.append({"trading_date": r["trading_date"], "candidate": r["candidate"], "version": r["version"], "same": bool(same)})
    return {"reproduce_ok": all(o["same"] for o in out), "checks": out}
