"""매매 시뮬레이션 실행 (5·9·10·12절).

  python run_backtest.py                    # 최신 daily_decision_log 를 읽어 매매
  python run_backtest.py --gate-run <id>    # 특정 게이트 실행 결과로
  python run_backtest.py --synthetic        # 합성 데이터 (run_gate.py --synthetic 다음)

입력: logs/daily_decision_log/run_<id>.parquet (run_gate.py 출력), 분봉·오더북·펀딩 원자료
출력: logs/{backtest_decision_log, trade_log, signal_log, funding_ledger, daily_pnl_log}/run_<id>.parquet
      reports/backtest_summary_<id>.csv, reports/backtest_audit_<id>.json

게이트를 통과하지 못한 날도 같은 규칙으로 가상 매매를 계산해 둔다 (is_real=False).
13절 "게이트 정보력"(통과일 대 미통과일 가상 성과) 검정에 쓰인다. 실제 성과(daily_pnl_log.net_pnl)에는
is_real=True 인 거래만 들어간다.
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
import zlib
from datetime import date
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import config, engine, ingest, logger, thresholds
from src.funding import FundingSchedule, load_funding
from src.gate_runner import date_range
from src.sessions import build_sessions, holding_segments, segment_layout, trading_day_bounds
from src import estar
from src.gate_runner import slice_window
from src.tradecal import TradingCalendar

KST = "Asia/Seoul"


def load_market(cfg, proc_dir, mid_dir, syms):
    px = ingest.load_minute_prices(proc_dir, syms, mid_dir)
    opens, closes = {}, {}
    for key, sym in syms.items():
        p = os.path.join(proc_dir, f"bars_1m_{sym}.parquet")
        g = pd.read_parquet(p).set_index("ts_utc")
        closes[key] = g["close"]
        opens[key] = g["open"] if "open" in g else g["close"].shift(1)
    books = {}
    if mid_dir:
        for key, sym in syms.items():
            p = os.path.join(mid_dir, f"mid_1m_{sym}.parquet")
            if os.path.exists(p):
                b = pd.read_parquet(p)
                b = b[b["valid"]].set_index("ts_utc")[["bid_px", "bid_qty", "ask_px", "ask_qty"]]
                books[key] = b
    return px, pd.DataFrame(opens), pd.DataFrame(closes), books


def job_seed(cfg, d: str, cand: str) -> int:
    return int(cfg["thresholds"]["seed"]) + int(d.replace("-", "")) + zlib.crc32(cand.encode()) % 1000


def model_for_rule(cfg, rule: str) -> str:
    """KSS 버전은 config 의 비선형 임계값 방식, 나머지는 선형."""
    if rule == "kss" and cfg["trading"].get("nonlinear_thresholds", "linear") == "estar":
        return "estar"
    return "block_bootstrap" if cfg["thresholds"].get("linear_path_method", "parametric") == "block_bootstrap" else "linear"


def day_z_stats(cal, px, r) -> dict:
    """거래일 [T0, 07:59] 의 z 를 윈도우 모수로 계산한 평균·표준편차 (표본 밖 수준 이동 측정용)."""
    t0, last = trading_day_bounds(cal, date.fromisoformat(r["trading_date"]))
    w = px.loc[(px.index >= t0.tz_convert("UTC")) & (px.index <= last.tz_convert("UTC")), ["L", "A"]].dropna()
    if len(w) < 60:
        return {"n": len(w), "mean": np.nan, "std": np.nan}
    bb = 1.0 if r["candidate"] == "struct" else r["b"]
    z = ((np.log(w["A"]) - bb * np.log(w["L"]) - r["a"]) - r["mu"]) / r["sigma"]
    return {"n": len(w), "mean": float(z.mean()), "std": float(z.std())}


def level_sd_for(cfg, day_stats: dict, d: str, cand: str) -> tuple[float, int]:
    """T0 이전 거래일들(같은 후보)의 그날 z 평균 RMS. 거래일 d-1 은 d 의 T0 직전에 끝나므로 룩어헤드 없음."""
    tc = cfg["thresholds"]
    if not tc.get("level_shift", False):
        return 0.0, 0
    past = sorted((dd, v["mean"]) for (dd, c), v in day_stats.items() if c == cand and dd < d and np.isfinite(v["mean"]))
    past = past[-int(tc.get("level_shift_lookback_days", 10)):]
    if len(past) < int(tc.get("level_shift_min_obs", 3)):
        return float(tc.get("level_shift_sd_default", 1.0)), len(past)
    m = np.array([v for _, v in past])
    return float(np.sqrt(np.mean(m * m))), len(past)


def run_one(job):
    cfg, d, cand, params, sessions, md, sched, day_minutes = (
        job["cfg"], job["date"], job["cand"], job["params"], job["sessions"], job["md"], job["sched"], job["minutes"])
    model = job["model"]
    seed = job_seed(cfg, d, cand)
    fc = cfg["funding"]
    rate_fc = sched.forecast(job["t0"], int(fc.get("forecast_lookback", 6)), fc.get("forecast_method", "signed")) \
        if fc.get("include_in_thresholds", False) else None
    layout = segment_layout(sessions, sched)
    est = {"estar_phi": np.nan, "estar_gamma": np.nan, "estar_lags": -1, "estar_speed_at_2": np.nan, "estar_reason": "",
           "estar_delta": np.nan}
    thr = None
    lsd = job.get("level_sd", 0.0)
    zs = None
    if "window" in job:
        w = job["window"]
        zs = estar.window_z(w["lnA"], w["lnL"], w["idx"], params["a"], params["b"], params["mu"], params["sigma"],
                            cand == "struct", int(cfg["data_quality"].get("max_gap_min", 5)))
    if model == "estar":
        fit = estar.fit(zs, int(cfg["thresholds"].get("estar_max_lags", 10)))
        est.update({"estar_phi": fit.get("phi", np.nan), "estar_gamma": fit.get("gamma", np.nan),
                    "estar_lags": fit.get("p", -1), "estar_speed_at_2": fit.get("speed_at_2", np.nan),
                    "estar_reason": fit.get("reason", ""), "estar_delta": fit.get("delta", np.nan)})
        if fit["ok"]:
            thr = thresholds.optimize(cfg, params, params["b"], cand, layout, md.symbols, seed,
                                      model="estar", estar_model=fit, rate_fc=rate_fc, level_sd=lsd)
            if not thr["ok"]:
                est["estar_reason"] = thr["reason"]
                thr = None
        if thr is None:
            fb = "block_bootstrap" if zs else "linear"
            thr = thresholds.optimize(cfg, params, params["b"], cand, layout, md.symbols, seed, model=fb,
                                      rate_fc=rate_fc, z_segs=zs, level_sd=lsd)
            if thr["ok"]:
                thr["model"] = f"{fb}_fallback"
    else:
        thr = thresholds.optimize(cfg, params, params["b"], cand, layout, md.symbols, seed, model=model,
                                  rate_fc=rate_fc, z_segs=zs, level_sd=lsd)
    thr.update(est)
    thr["seed"] = seed
    thr["rate_fc"] = rate_fc or {}
    thr["level_sd_obs"] = job.get("level_sd_obs", 0)
    out = {"date": d, "cand": cand, "model": model, "thr": thr}
    if not thr["ok"]:
        out["res"] = {"trades": [], "signals": [], "funding": []}
        return out
    out["res"] = engine.run_day(cfg, md, sessions, params, thr, cand, sched, day_minutes, d)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--gate-run")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args(argv)

    cfg = config.load_config(a.config)
    syms = {"L": cfg["symbols"]["local"], "A": cfg["symbols"]["adr"]}
    log_dir = config.abspath(cfg, "log_dir")
    report_dir = config.abspath(cfg, "report_dir")
    if a.synthetic:
        log_dir = os.path.join(log_dir, "synthetic")
        proc = os.path.join(config.ROOT, "data", "synthetic", "processed")
        mid_dir = None
        funding = load_funding("/nonexistent")
    else:
        proc = config.abspath(cfg, "processed_dir")
        mid_dir = config.abspath(cfg, "mid_dir")
        funding = load_funding(os.path.join(config.ROOT, cfg["funding"]["raw_dir"]))

    gdir = os.path.join(log_dir, "daily_decision_log")
    if a.gate_run:
        gpath = os.path.join(gdir, f"run_{a.gate_run}.parquet")
    else:
        gpath = sorted(glob.glob(os.path.join(gdir, "run_*.parquet")))[-1]
    gate = pd.read_parquet(gpath)
    gate_run_id = gate["run_id"].iloc[0]
    print(f"게이트 입력: {os.path.basename(gpath)}  ({len(gate)} 행)")

    cal = TradingCalendar(cfg)
    sched = FundingSchedule(cfg, funding)
    print(f"정산 시각표: {sched.summary()}")
    px, opens, closes, books = load_market(cfg, proc, mid_dir, syms)

    usable = gate[gate["stats_computed"] & gate["a"].notna() & gate["sigma"].notna()]
    usable = usable[(usable["b"] > 0) | (usable["candidate"] == "struct")]
    usable = usable.assign(model=[model_for_rule(cfg, r) for r in usable["gate_rule"]])
    keys = usable.drop_duplicates(["trading_date", "candidate", "model"])
    day_stats = {(r["trading_date"], r["candidate"]): day_z_stats(cal, px, r)
                 for _, r in usable.drop_duplicates(["trading_date", "candidate"]).iterrows()}
    jobs = []
    win_cache = {}
    for _, r in keys.iterrows():
        d = date.fromisoformat(r["trading_date"])
        sessions = build_sessions(cal, sched, d, cfg["funding"]["buffer_sec"], cfg["funding"].get("policy", "exit"))
        t0, last = trading_day_bounds(cal, d)
        lo, hi = t0.tz_convert("UTC") - pd.Timedelta(minutes=1), last.tz_convert("UTC") + pd.Timedelta(minutes=2)
        day_minutes = list(pd.date_range(t0, last, freq="1min").tz_convert("UTC"))
        sl = lambda df: df.loc[(df.index >= lo) & (df.index <= hi)]
        md = engine.MarketDay(sl(px), sl(opens), sl(closes), {k: sl(v) for k, v in books.items()}, syms)
        params = {k: float(r[k]) for k in ("a", "b", "mu", "sigma", "phi", "phi_ar1", "half_life_min")}
        job = {"cfg": cfg, "date": r["trading_date"], "cand": r["candidate"], "params": params, "model": r["model"],
               "sessions": sessions, "md": md, "sched": sched, "minutes": day_minutes, "t0": t0}
        job["level_sd"], job["level_sd_obs"] = level_sd_for(cfg, day_stats, r["trading_date"], r["candidate"])
        if r["model"] != "linear":
            wk = (r["trading_date"], r["version"])
            if wk not in win_cache:
                wd = slice_window(px, cal.window(d, r["version"]), cfg)
                win_cache[wk] = {"lnA": wd["lnA"], "lnL": wd["lnL"], "idx": wd["idx_used"]}
            job["window"] = win_cache[wk]
        jobs.append(job)
    print(f"계산 대상 (거래일 x 후보 x 임계값 모형): {len(jobs)}")
    t = time.time()
    if a.processes > 1 and len(jobs) > 1:
        with Pool(a.processes) as p:
            results = p.map(run_one, jobs)
    else:
        results = [run_one(j) for j in jobs]
    print(f"계산 {time.time() - t:.0f}초")
    res_by = {(r["date"], r["cand"], r["model"]): r for r in results}
    sess_by = {(j["date"], j["cand"]): j["sessions"] for j in jobs}

    run_id = logger.new_run_id()
    meta = {"run_id": run_id, "gate_run_id": gate_run_id, "config_hash": cfg["_config_hash"],
            "code_version": config.code_version(), "data_snapshot_id": gate["data_snapshot_id"].iloc[0]}
    C = cfg["trading"]["capital_usd"]
    regime = lambda ts: cal.regime(pd.Timestamp(ts).tz_convert(KST))

    dec_rows, trade_rows, sig_rows, fund_rows = [], [], [], []
    for _, g in gate.iterrows():
        key = (g["trading_date"], g["candidate"])
        rr = res_by.get(key + (model_for_rule(cfg, g["gate_rule"]),))
        thr = rr["thr"] if rr else {"ok": False, "reason": "NOT_COMPUTED"}
        res = rr["res"] if rr else {"trades": [], "signals": [], "funding": []}
        gate_trade = g["decision"] == "TRADE"
        if gate_trade and thr.get("ok") and thr["exp_pnl_usd"] <= 0:
            final, reason = "NO_TRADE", "COST_FILTER"
        elif gate_trade and not thr.get("ok"):
            final, reason = "NO_TRADE", thr.get("reason", "BAD_PARAMS")
        elif gate_trade:
            final, reason = "TRADE", ""
        else:
            final, reason = "NO_TRADE", g["no_trade_reason"]
        is_real = final == "TRADE"
        sess = sess_by.get(key, [])
        dec_rows.append({
            **meta, "trading_date": g["trading_date"], "candidate": g["candidate"], "gate_rule": g["gate_rule"],
            "version": g["version"], "gate_decision": g["decision"], "gate_reason": g["no_trade_reason"],
            **{k: g[k] for k in ("a", "b", "mu", "sigma", "phi", "half_life_min")},
            "n_sessions": len(sess), "session_lengths": ",".join(str(s.minutes) for s in sess),
            "funding_policy": cfg["funding"].get("policy", "exit"),
            "holding_segments": ",".join(str(x) for x in holding_segments(sess)) if sess else "",
            "settlement_schedule": json.dumps({s: f"{i['interval_h']}h/{i['source']}" for s, i in sched.info.items()}),
            "entry_z": thr.get("entry_z", np.nan), "exit_z": thr.get("exit_z", np.nan), "stop_z": thr.get("stop_z", np.nan),
            "bertram_entry_z": thr.get("bertram_entry_z", np.nan), "exp_pnl_usd": thr.get("exp_pnl_usd", np.nan),
            "exp_trades": thr.get("exp_trades", np.nan), "cost_roundtrip_z": thr.get("cost_roundtrip_z", np.nan),
            "latent_var_share": thr.get("latent_var_share", np.nan), "threshold_seed": thr.get("seed", -1),
            "threshold_model": thr.get("model", ""), "estar_phi": thr.get("estar_phi", np.nan),
            "estar_gamma": thr.get("estar_gamma", np.nan), "estar_delta": thr.get("estar_delta", np.nan), "estar_lags": thr.get("estar_lags", -1),
            "estar_speed_at_2": thr.get("estar_speed_at_2", np.nan), "estar_reason": thr.get("estar_reason", ""),
            "exp_funding_usd": thr.get("exp_funding_usd", np.nan),
            "level_sd": thr.get("level_sd", np.nan), "level_sd_obs": thr.get("level_sd_obs", 0),
            "day_z_mean": day_stats.get(key, {}).get("mean", np.nan), "day_z_std": day_stats.get(key, {}).get("std", np.nan),
            "funding_fc_L": thr.get("rate_fc", {}).get(syms["L"], np.nan),
            "funding_fc_A": thr.get("rate_fc", {}).get(syms["A"], np.nan),
            "final_decision": final, "final_reason": reason, "is_real": is_real,
            "n_trades": len(res["trades"]), "net_pnl": float(sum(tr["net_pnl"] for tr in res["trades"])),
        })
        for i, tr in enumerate(res["trades"]):
            trade_rows.append({**meta, **tr, "gate_rule": g["gate_rule"], "is_real": is_real, "trade_no": i,
                               "entry_signal_kst": str(pd.Timestamp(tr["entry_signal_utc"]).tz_convert(KST)),
                               "exit_signal_kst": str(pd.Timestamp(tr["exit_signal_utc"]).tz_convert(KST)),
                               "entry_regime": regime(tr["entry_signal_utc"]), "exit_regime": regime(tr["exit_signal_utc"]),
                               "weekend_flag": pd.Timestamp(tr["entry_signal_utc"]).tz_convert(KST).weekday() >= 5})
        seen_ep = set()
        for sg in res["signals"]:
            first = sg["episode_id"] not in seen_ep
            seen_ep.add(sg["episode_id"])
            skip = sg["skip_reason"]
            if not is_real and sg["executed"]:
                skip = reason or "GATE_FAIL"
            sig_rows.append({**meta, **sg, "gate_rule": g["gate_rule"], "is_first_in_episode": first,
                             "minute_kst": str(pd.Timestamp(sg["minute_utc"]).tz_convert(KST)),
                             "regime": regime(sg["minute_utc"]),
                             "executed": bool(sg["executed"] and is_real),
                             "hypothetical_executed": bool(sg["executed"]), "skip_reason": skip})
        for f in res["funding"]:
            fund_rows.append({**meta, **f, "gate_rule": g["gate_rule"], "is_real": is_real})

    dec = pd.DataFrame(dec_rows)
    trades = pd.DataFrame(trade_rows)
    # 일별 손익: 표본 기간 모든 달력일 x 후보 x 규칙
    d0, d1 = date.fromisoformat(gate["trading_date"].min()), date.fromisoformat(gate["trading_date"].max())
    pnl_rows = []
    for d in date_range(d0, d1):
        ds = d.isoformat()
        for cand in ("ols", "tls", "dols", "struct"):
            for rule in sorted(gate["gate_rule"].unique()):
                dr = dec[(dec.trading_date == ds) & (dec.candidate == cand) & (dec.gate_rule == rule)]
                tr = trades[(trades.trading_date == ds) & (trades.candidate == cand) & (trades.gate_rule == rule)] \
                    if len(trades) else trades
                real = bool(len(dr)) and bool(dr["is_real"].iloc[0])
                status = (dr["final_decision"].iloc[0] if real else dr["final_reason"].iloc[0]) if len(dr) else "NO_DATA"
                s = lambda col: float(tr[col].sum()) if len(tr) else 0.0
                pnl_rows.append({**meta, "trading_date": ds, "candidate": cand, "gate_rule": rule, "status": status,
                                 "capital_usd": C,
                                 "n_trades": len(tr) if real else 0, "net_pnl": s("net_pnl") if real else 0.0,
                                 "ret": (s("net_pnl") if real else 0.0) / C,
                                 "ret_cost1_5": (s("net_pnl_cost1_5") if real else 0.0) / C,
                                 "ret_cost2": (s("net_pnl_cost2") if real else 0.0) / C,
                                 "ret_fee_alt": (s("net_pnl_fee_alt") if real else 0.0) / C,
                                 "hypo_n_trades": len(tr), "hypo_net_pnl": s("net_pnl"), "hypo_ret": s("net_pnl") / C,
                                 "hypo_net_pnl_fee_alt": s("net_pnl_fee_alt")})
    paths = {}
    for name, rows in (("backtest_decision_log", dec_rows), ("trade_log", trade_rows), ("signal_log", sig_rows),
                       ("funding_ledger", fund_rows), ("daily_pnl_log", pnl_rows)):
        if rows:
            paths[name] = logger.write_log(rows, name, log_dir, run_id)
        else:
            cols = list(logger.load_schema(name)["columns"])
            os.makedirs(os.path.join(log_dir, name), exist_ok=True)
            pd.DataFrame(columns=cols).to_parquet(os.path.join(log_dir, name, f"run_{run_id}.parquet"), index=False)

    # ---- 감사 (12절)
    pnl = pd.DataFrame(pnl_rows)
    aud = {}
    if len(trades):
        real_tr = trades[trades.is_real].groupby(["trading_date", "candidate", "gate_rule"])["net_pnl"].sum()
        pr = pnl.set_index(["trading_date", "candidate", "gate_rule"])["net_pnl"]
        aud["aggregate_ok"] = bool(np.allclose(real_tr.reindex(pr.index).fillna(0).values, pr.values))
        fsum = sum(f["amount"] for f in fund_rows)
        aud["funding_ok"] = bool(np.isclose(-fsum, trades["funding"].sum())) and len(fund_rows) == int(trades["funding_events"].sum())
        aud["funding_events"] = int(trades["funding_events"].sum())
        aud["funding_missing_rate"] = int(trades["funding_missing"].sum())
        aud["funding_cost_total"] = float(trades["funding"].sum())
        aud["forced_exits"] = trades["exit_type"].value_counts().to_dict()
        aud["data_gap_exits"] = int(trades["flags"].str.contains("DATA_GAP").sum())
    else:
        aud.update({"aggregate_ok": True, "funding_ok": True, "funding_events": 0, "note": "거래 없음"})
    aud["settlement_schedule"] = sched.summary()
    with open(os.path.join(report_dir, f"backtest_audit_{run_id}.json"), "w") as f:
        json.dump(aud, f, ensure_ascii=False, indent=1, default=str)
    summ = pnl.groupby(["candidate", "gate_rule"]).agg(
        days=("trading_date", "count"), real_days=("status", lambda s: int((s == "TRADE").sum())),
        n_trades=("n_trades", "sum"), net_pnl=("net_pnl", "sum"),
        hypo_trades=("hypo_n_trades", "sum"), hypo_net_pnl=("hypo_net_pnl", "sum"),
        hypo_net_fee_alt=("hypo_net_pnl_fee_alt", "sum")).reset_index()
    summ.to_csv(os.path.join(report_dir, f"backtest_summary_{run_id}.csv"), index=False, encoding="utf-8-sig")
    # 임계값 시뮬레이션 보정 점검: 기대(시뮬레이션) vs 실현(가상 매매, 게이트 무시)
    cal_df = dec[dec["threshold_model"] != ""][["trading_date", "candidate", "gate_rule", "threshold_model", "level_sd",
                                                "exp_trades", "n_trades", "exp_pnl_usd", "net_pnl", "day_z_mean"]]
    cal_df.to_csv(os.path.join(report_dir, f"threshold_calibration_{run_id}.csv"), index=False, encoding="utf-8-sig")
    calib = cal_df.groupby("gate_rule")[["exp_trades", "n_trades", "exp_pnl_usd", "net_pnl"]].sum().round(1)
    print(f"run_id={run_id}  (정산 정책: {cfg['funding'].get('policy', 'exit')}, 대체 수수료 {cfg['costs'].get('fee_alt')})")
    print(summ.round(2).to_string(index=False))
    print("임계값 보정 점검 (기대 vs 가상 매매 실현, 합계):")
    print(calib.to_string())
    print(f"감사: {json.dumps({k: v for k, v in aud.items() if k != 'settlement_schedule'}, ensure_ascii=False, default=str)}")
    if not aud["aggregate_ok"] or not aud["funding_ok"]:
        sys.exit("감사 실패")
    return run_id


if __name__ == "__main__":
    main()
