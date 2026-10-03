"""검정 파이프라인 실행 스크립트.

  python run_gate.py                       # 원자료 data/raw -> 전처리 -> 전체 거래일 검정 -> 로그
  python run_gate.py --ignore-coverage     # 결측 10% 초과 윈도우도 통계량은 계산 (결정은 DATA_BAD 유지)
  python run_gate.py --synthetic           # 합성 데이터(data/synthetic)로 끝까지 작동 확인
  python run_gate.py --start 2026-09-20 --end 2026-09-25 --B 99

출력: logs/daily_decision_log/run_<run_id>.parquet, reports/gate_summary_<run_id>.csv,
      reports/audit_<run_id>.json, reports/data_quality.json
"""
from __future__ import annotations

import os

# 작은 행렬 연산이 많아 BLAS 다중 스레드가 오히려 4배 느리다. 병렬화는 거래일 단위 프로세스로.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time
from datetime import date

import pandas as pd

from src import audit, config, gate_runner, ingest, logger, orderbook


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--B", type=int)
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ignore-coverage", action="store_true")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--skip-reproduce", action="store_true")
    a = ap.parse_args(argv)

    cfg = config.load_config(a.config)
    if a.B:
        cfg["bootstrap"]["B"] = a.B
        cfg["_config_hash"] += f"+B{a.B}"
    syms = {"L": cfg["symbols"]["local"], "A": cfg["symbols"]["adr"]}
    report_dir = config.abspath(cfg, "report_dir")
    log_dir = config.abspath(cfg, "log_dir")
    os.makedirs(report_dir, exist_ok=True)

    if a.synthetic:
        proc = os.path.join(config.ROOT, "data", "synthetic", "processed")
        files = sorted(glob.glob(os.path.join(proc, "*.parquet")))
        if not files:
            sys.exit("합성 데이터가 없습니다: python make_synthetic.py 먼저 실행")
        snapshot = "SYNTH-" + config.data_snapshot_id(files)
        log_dir = os.path.join(log_dir, "synthetic")
    else:
        proc = config.abspath(cfg, "processed_dir")
        _, rep, files = ingest.run_ingest(cfg, config.abspath(cfg, "raw_dir"), proc, report_dir)
        snapshot = config.data_snapshot_id(files)
        if rep["missing_symbols"]:
            print(f"[경고] 원자료에 없는 심볼: {rep['missing_symbols']}  -> Step 2·3 불가, Step 1 은 있는 심볼만")

    mid_dir = None
    if not a.synthetic:
        ob_dir = config.abspath(cfg, "raw_orderbook_dir") if "raw_orderbook_dir" in cfg["paths"] else None
        if ob_dir and glob.glob(os.path.join(ob_dir, "*.parquet")):
            mid_dir = config.abspath(cfg, "mid_dir")
            ob_rep = orderbook.build_mid_files(ob_dir, mid_dir)
            print(f"오더북 재구성: {ob_rep}")
            files = files + sorted(glob.glob(os.path.join(ob_dir, "*.parquet")))
            snapshot = config.data_snapshot_id(files)
    px = ingest.load_minute_prices(proc, syms, mid_dir)
    first = px.index.min().tz_convert("Asia/Seoul").date()
    last = px.index.max().tz_convert("Asia/Seoul").date()
    start = date.fromisoformat(a.start) if a.start else first
    end = date.fromisoformat(a.end) if a.end else last
    dates = list(gate_runner.date_range(start, end))

    run_id = logger.new_run_id()
    meta = {"run_id": run_id, "config_hash": cfg["_config_hash"], "code_version": config.code_version(),
            "data_snapshot_id": snapshot}
    print(f"run_id={run_id}  거래일 {start}~{end} ({len(dates)}일)  B={cfg['bootstrap']['B']}  "
          f"processes={a.processes}  ignore_coverage={a.ignore_coverage}")
    t = time.time()
    rows = gate_runner.run(cfg, px, dates, meta, ignore_coverage=a.ignore_coverage, processes=a.processes)
    print(f"계산 {time.time() - t:.0f}초")
    path = logger.write_log(rows, "daily_decision_log", log_dir, run_id)
    df = pd.read_parquet(path)

    aud = audit.audit_log(df)
    if not a.skip_reproduce:
        fn = lambda c, p, ds: gate_runner.run(c, p, ds, meta, ignore_coverage=a.ignore_coverage, processes=1)
        aud.update(audit.reproduce_check(cfg, px, dates, fn, df))
    with open(os.path.join(report_dir, f"audit_{run_id}.json"), "w") as f:
        json.dump(aud, f, ensure_ascii=False, indent=1)

    cols = ["trading_date", "weekday", "version", "candidate", "day_type", "window_flags", "bars_paired",
            "missing_frac_paired", "data_flags", "step1_pass", "step1_reason", "b", "adf_stat", "adf_p",
            "kss_stat", "kss_p", "gate_class", "half_life_min", "decision", "no_trade_reason"]
    summ = df[cols]
    summ.to_csv(os.path.join(report_dir, f"gate_summary_{run_id}.csv"), index=False, encoding="utf-8-sig")
    print(f"로그: {path}")
    print(f"감사: causality_ok={aud['causality_ok']} unique_key_ok={aud['unique_key_ok']} "
          f"reproduce_ok={aud.get('reproduce_ok')}")
    for w in aud["warnings"]:
        print("  [경고]", w)
    if not aud["causality_ok"] or aud.get("reproduce_ok") is False or not aud["unique_key_ok"]:
        sys.exit("감사 실패")
    return df, aud


if __name__ == "__main__":
    main()
