"""거래일 x 버전 단위로 윈도우를 잘라 Step 1~4 를 돌리고 daily_decision_log 행을 만든다.

같은 윈도우(토요일 재사용 등)는 한 번만 계산하고 결과를 재사용한다.
"""
from __future__ import annotations

import json
import zlib
from datetime import date, timedelta
from multiprocessing import Pool

import numpy as np
import pandas as pd

from . import steps
from .tradecal import TradingCalendar, Window

MIN_BARS = 240   # 기본값. config [data_quality] min_bars 가 우선
DATA_COLS = ("bars_L", "bars_A", "bars_paired", "bars_used", "segments_used", "longest_segment_min",
             "missing_frac_L", "missing_frac_A", "missing_frac_paired", "backfill_bars",
             "late_receive_bars", "late_receive_max_sec", "last_bar_excluded")
PARAM_COLS = ("mu", "sigma", "phi", "theta", "half_life_min", "phi_ar1", "half_life_ar1_min",
              "phi_arma", "ma_arma", "half_life_arma_min")


def _price_type(mid_frac) -> str:
    if mid_frac is None or not np.isfinite(mid_frac) or mid_frac == 0:
        return "trade_close"
    return "mid" if mid_frac >= 1 else f"mixed_mid{mid_frac:.0%}"


def is_bad(data: dict, cfg: dict) -> bool:
    """DATA_BAD 판정. ADR 없음, 너무 짧음, (규칙 사용 시) 결측 비율 초과."""
    dq = cfg["data_quality"]
    if data["bars_A"] == 0 or data["bars_used"] < dq.get("min_bars", MIN_BARS):
        return True
    if dq.get("enforce_missing_rule", True):
        m = dq["max_missing_frac"]
        return data["missing_frac_paired"] > m or data["missing_frac_L"] > m
    return False


def rng_factory_for(seed: int):
    def f(name: str):
        return np.random.default_rng([seed, zlib.crc32(name.encode())])
    return f


def day_seed(cfg: dict, d: date) -> int:
    return int(cfg["bootstrap"]["base_seed"]) + int(d.strftime("%Y%m%d"))


def segments_of(idx: pd.DatetimeIndex, max_gap_min: int) -> list[tuple[int, int]]:
    """연속 구간 [i0, i1] (위치 인덱스). 인접 분 간격이 max_gap_min 이하이면 같은 구간."""
    if len(idx) == 0:
        return []
    gap = np.diff(idx.asi8) // 60_000_000_000
    cuts = np.where(gap > max_gap_min)[0]
    starts = np.r_[0, cuts + 1]
    ends = np.r_[cuts, len(idx) - 1]
    return list(zip(starts.tolist(), ends.tolist()))


def slice_window(px: pd.DataFrame, win: Window, cfg: dict | None = None) -> dict:
    dq = (cfg or {}).get("data_quality", {})
    s, e = win.start.tz_convert("UTC"), win.end.tz_convert("UTC")
    excl = bool(dq.get("exclude_last_bar", False))
    if excl:   # 07:59 분봉은 T0 직후에 도착하므로 제외 (2026-10-03 결정)
        e = e - pd.Timedelta(minutes=1)
    w = px.loc[(px.index >= s) & (px.index < e)]
    exp = max(1, win.expected_minutes - (1 if excl else 0))
    L, A = w["L"], w["A"]
    paired_all = w[L.notna() & A.notna()]
    # 최소 연속 구간 기준: 간격 max_gap_min 이하로 이어진 구간 중 길이 >= min_segment_min 인 것만 사용
    segs = segments_of(paired_all.index, int(dq.get("max_gap_min", 5)))
    seg_len = [int((paired_all.index[j] - paired_all.index[i]) / pd.Timedelta(minutes=1)) + 1 for i, j in segs]
    min_seg = int(dq.get("min_segment_min", 0))
    keep = [k for k, n in enumerate(seg_len) if n >= min_seg]
    pos = np.concatenate([np.arange(segs[k][0], segs[k][1] + 1) for k in keep]) if keep else np.array([], int)
    paired = paired_all.iloc[pos]
    mid_frac = np.nan
    if "L_is_mid" in paired and len(paired):
        # 두 다리 가격 중 중간가의 비율 (0 = 모두 체결가, 1 = 모두 중간가)
        mid_frac = float((paired["L_is_mid"].fillna(False).astype(float) + paired["A_is_mid"].fillna(False).astype(float)).mean() / 2)
    t0_ns = win.t0.tz_convert("UTC").value
    late = []
    for k in ("L", "A"):
        col = f"{k}_avail_ts_ns"
        if col in w:
            v = w[col].dropna()
            late.append(v[v > t0_ns])
    late = pd.concat(late) if late else pd.Series(dtype=float)
    bf = 0
    for k in ("L", "A"):
        col = f"{k}_backfill"
        if col in w:
            bf += int(w[col].fillna(False).astype(bool).sum())
    return {
        "lnL": np.log(paired["L"].to_numpy()) if len(paired) else np.array([]),
        "lnA": np.log(paired["A"].to_numpy()) if len(paired) else np.array([]),
        "lnL_only": np.log(L.dropna().to_numpy()),
        "bars_L": int(L.notna().sum()), "bars_A": int(A.notna().sum()), "bars_paired": int(len(paired_all)),
        "bars_used": int(len(paired)), "segments_used": int(len(keep)),
        "longest_segment_min": int(max(seg_len)) if seg_len else 0,
        "mid_frac": mid_frac, "last_bar_excluded": excl,
        "missing_frac_L": 1 - L.notna().sum() / exp, "missing_frac_A": 1 - A.notna().sum() / exp,
        "missing_frac_paired": 1 - len(paired_all) / exp,
        "backfill_bars": bf,
        "late_receive_bars": int(len(late)),
        "late_receive_max_sec": float((late.max() - t0_ns) / 1e9) if len(late) else 0.0,
        "last_bar_open_utc": str(paired.index.max()) if len(paired) else "",
        "idx_used": paired.index,          # 매매 단계(ESTAR 적합)에서 연속 구간을 다시 나눌 때 사용
    }


def compute_window(job: dict) -> dict:
    """프로세스 풀에서 실행. job: cfg, seed, data(slice), compute(bool)."""
    cfg, seed, data = job["cfg"], job["seed"], job["data"]
    rf = rng_factory_for(seed)
    res = {"step1": None, "step3": None, "params": {}}
    if not job["compute"]:
        return res
    have_A = data["bars_A"] > 0
    if have_A and data["bars_used"] >= cfg["data_quality"].get("min_bars", MIN_BARS):
        res["step1"] = steps.step1(data["lnL"], data["lnA"], cfg, rf)
        s3 = steps.step3_all(data["lnA"], data["lnL"], cfg, rf)
        for cand, r in s3.items():
            res["params"][cand] = steps.step4_params(r.pop("spread"), cfg["params"].get("half_life_method", "arma11"))
        res["step3"] = s3
    elif len(data["lnL_only"]) >= cfg["data_quality"].get("min_bars", MIN_BARS):
        # ADR 없음: 본주만 Step 1 (진단용)
        res["step1"] = steps.step1(data["lnL_only"], None, cfg, rf)
    return res


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items() if k != "draws"}
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def build_jobs(cfg, px, dates, ignore_coverage=False):
    cal = TradingCalendar(cfg)
    maxmiss = cfg["data_quality"]["max_missing_frac"]
    entries, jobs, seen = [], [], {}
    for d in dates:
        for v in cfg["calendar"].get("versions", ["V1", "V2"]):
            win = cal.window(d, v)
            key = (win.start, win.end, v)
            if win.expected_minutes == 0:
                entries.append((d, v, win, None, None, None))
                continue
            if key in seen:
                entries.append((d, v, win, seen[key], None, None))
                continue
            data = slice_window(px, win, cfg)
            bad = is_bad(data, cfg)
            compute = (not bad) or ignore_coverage
            seed = day_seed(cfg, d)
            jobs.append({"cfg": cfg, "seed": seed, "data": data, "compute": compute, "key": key})
            seen[key] = d
            entries.append((d, v, win, None, data, (bad, seed, len(jobs) - 1)))
    return cal, entries, jobs


def run(cfg, px, dates, meta, ignore_coverage=False, processes=2):
    cal, entries, jobs = build_jobs(cfg, px, dates, ignore_coverage)
    if processes > 1 and len(jobs) > 1:
        with Pool(processes) as pool:
            results = pool.map(compute_window, jobs)
    else:
        results = [compute_window(j) for j in jobs]
    by_key = {j["key"]: (j, r) for j, r in zip(jobs, results)}

    rows = []
    for d, v, win, reused_from, data, info in entries:
        key = (win.start, win.end, v)
        if key in by_key:
            job, res = by_key[key]
            data = job["data"]
            seed = job["seed"]
        else:
            job, res, seed = None, {"step1": None, "step3": None, "params": {}}, None
        maxmiss = cfg["data_quality"]["max_missing_frac"]
        flags = []
        if data is None:
            data = {k: np.nan for k in DATA_COLS}
            data["mid_frac"] = np.nan
            bad = True
        else:
            bad = is_bad(data, cfg)
            if data["bars_A"] == 0:
                flags.append("ADR_MISSING")
            if data["missing_frac_paired"] > maxmiss:
                flags.append("COVERAGE_BELOW_90PCT")
            if data["bars_used"] < cfg["data_quality"].get("min_bars", MIN_BARS) and data["bars_A"] > 0:
                flags.append("NO_LONG_SEGMENT" if data["bars_paired"] >= cfg["data_quality"].get("min_bars", MIN_BARS)
                             else "TOO_SHORT")
            if data["bars_used"] < data["bars_paired"]:
                flags.append("SHORT_SEGMENTS_DROPPED")
            if bad and ignore_coverage:
                flags.append("DIAGNOSTIC_OVERRIDE")
            if data["backfill_bars"]:
                flags.append("HAS_BACKFILL")
            if data["bars_paired"] and data["bars_paired"] < win.expected_minutes:
                flags.append("GAPS_DROPPED")
        s1 = res["step1"]
        step1_stats = json.dumps(_jsonable(s1["series"]), ensure_ascii=False) if s1 else ""
        for cand, rule in [(c, g) for c in ("ols", "tls", "dols", "struct")
                           for g in cfg["step3"].get("gate_rules", ["or"])]:
            r = (res["step3"] or {}).get(cand)
            p = res["params"].get(cand, {})
            row = {
                "gate_rule": rule,
                **meta, "random_seed": seed, "B": cfg["bootstrap"]["B"], "lag_criterion": cfg["lags"]["criterion"],
                "window_mode": cfg["calendar"].get("window_mode", "blocks"),
                "step1_pvalue_mode": cfg["step1"].get("pvalue_mode", "plan"),
                "step3_pvalue_mode": cfg["step3"].get("pvalue_mode", "plan"),
                "step1_gate_conditions": cfg["step1"].get("gate_conditions", "abcd"),
                "trading_date": d.isoformat(), "weekday": d.strftime("%a"), "candidate": cand, "version": v,
                "day_type": win.day_type,
                "t0_kst": str(win.t0), "t0_utc": str(win.t0.tz_convert("UTC")),
                "window_start_kst": str(win.start), "window_end_kst": str(win.end),
                "window_start_utc": str(win.start.tz_convert("UTC")), "window_end_utc": str(win.end.tz_convert("UTC")),
                "window_blocks": ",".join(win.blocks), "window_flags": ",".join(win.flags),
                "reused_window_from": reused_from.isoformat() if reused_from else "",
                "expected_minutes": win.expected_minutes,
                **{k: data[k] for k in DATA_COLS},
                "mid_frac": data.get("mid_frac", np.nan),
                "price_type": _price_type(data.get("mid_frac", np.nan)), "data_flags": ",".join(flags),
                "stats_computed": bool(s1 is not None),
                "step1_pass": bool(s1["pass"]) if s1 else False,
                "step1_reason": s1["reason"] if s1 else "",
                "step1_stats": step1_stats,
            }
            if r:
                row.update({
                    "a": r["a"], "b": r["b"], "b_ci_lo": r["b_ci_lo"], "b_ci_hi": r["b_ci_hi"],
                    "b_minus_1": r["b"] - 1, "dols_K": r["extra"].get("dols_K", np.nan),
                    "adf_stat": r["adf"]["stat"], "adf_lag": r["adf"]["k"], "adf_p": r["adf_p"],
                    "adf_p_source": r["adf_p_source"], "adf_p_boot": r["adf_p_boot"], "adf_p_table": r["adf_p_table"],
                    "adf_cv_boot": r["adf_cv_boot"], "kss_stat": r["kss"]["stat"], "kss_lag": r["kss"]["k"],
                    "kss_p": r["kss_p"], "kss_p_source": r.get("kss_p_source", "bootstrap"),
                    "kss_p_boot": r["kss_p_boot"], "kss_p_table": r.get("kss_p_table", np.nan),
                    "kss_cv_boot": r["kss_cv_boot"], "sieve_order": r["var_p"],
                    "adf_reject": r["adf_reject"], "kss_reject": r["kss_reject"],
                    "minp_p": r["minp_p"],
                    "gate_pass": r["gates"][rule][0], "gate_class": r["gates"][rule][1],
                    "nonlinear_flag": r["gates"][rule][1] == "NONLINEAR",
                    **{k: p[k] for k in PARAM_COLS}, "half_life_method": p["half_life_method"],
                    "half_life_ok": bool(p["half_life_min"] <= cfg["params"]["max_half_life_min"]),
                })
            else:
                for k in ("a", "b", "b_ci_lo", "b_ci_hi", "b_minus_1", "dols_K", "adf_stat", "adf_lag", "adf_p",
                          "adf_p_boot", "adf_p_table", "adf_cv_boot", "kss_stat", "kss_lag", "kss_p",
                          "kss_p_boot", "kss_p_table",
                          "kss_cv_boot", "sieve_order", "minp_p") + PARAM_COLS:
                    row[k] = np.nan
                row["half_life_method"] = ""
                row.update({"adf_p_source": "", "kss_p_source": "", "adf_reject": False, "kss_reject": False, "gate_pass": False,
                            "gate_class": "", "nonlinear_flag": False, "half_life_ok": False})
            # 결정 (비용 필터 전)
            if bad:
                reason = "DATA_BAD"
            elif "NO_NORMAL_BLOCK" in win.flags:
                reason = "HOLIDAY_RULE"
            elif not row["step1_pass"]:
                reason = "STEP1_UNDETERMINED"
            elif cand != "struct" and not (row["b"] > 0):
                reason = "BAD_HEDGE"
            elif not row["gate_pass"]:
                reason = "GATE_FAIL"
            elif not row["half_life_ok"]:
                reason = "HALFLIFE_FILTER"
            else:
                reason = ""
            row["decision"] = "NO_TRADE" if reason else "TRADE"
            row["no_trade_reason"] = reason
            row["cost_filter_applied"] = False
            rows.append(row)
    return rows


def date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)
