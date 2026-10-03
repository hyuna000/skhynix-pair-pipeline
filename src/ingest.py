"""3절 전처리: bar_1m 원자료 -> 심볼별 분봉 파일 + 품질 리포트.

- 가격·수량: Decimal 로 읽는다. 문자열이면 1e18 배율 정수로 보고 나눈다.
  (pyarrow decimal(…,18) 로 이미 들어온 경우는 그대로 Decimal.) 통계 계산용으로만 float 변환.
- event_id 중복 제거, closed=False 봉 제외, 시각순 정렬.
- 사용 가능 시각 avail_ts_ns: realtime 은 receive_ts_ns, backfill 은 봉 마감 시각(+표시).
- trading_date 는 KST 로 다시 계산해 원 필드와 대조.
"""
from __future__ import annotations

import glob
import json
import os
from decimal import Decimal

import numpy as np
import pandas as pd

SCALE = Decimal(10) ** 18
KST = "Asia/Seoul"
PRICE_COLS = ["open", "high", "low", "close", "volume"]


def to_decimal(v) -> Decimal | None:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, Decimal):
        return v
    if isinstance(v, (bytes, str)):
        s = v.decode() if isinstance(v, bytes) else v
        s = s.strip().strip('"')
        if "." in s:            # 이미 소수 표기
            return Decimal(s)
        return Decimal(s) / SCALE  # 1e18 배율 정수 문자열
    return Decimal(str(v))


def load_raw(raw_dir: str) -> tuple[pd.DataFrame, list[str]]:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet in {raw_dir}")
    df = pd.concat([pd.read_parquet(f).assign(_file=os.path.basename(f)) for f in files], ignore_index=True)
    return df, files


def clean_bars(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    stats = {"rows_raw": int(len(df))}
    df = df[df["event_type"].astype(str) == "bar_1m"].copy()
    stats["rows_bar_1m"] = int(len(df))
    n0 = len(df)
    df = df.drop_duplicates("event_id")
    stats["dup_event_id_removed"] = int(n0 - len(df))
    n0 = len(df)
    df = df.drop_duplicates(["instrument_raw_symbol", "open_ts_ns"], keep="last")
    stats["dup_symbol_open_ts_removed"] = int(n0 - len(df))
    if "closed" in df:
        stats["unclosed_removed"] = int((~df["closed"]).sum())
        df = df[df["closed"]]
    for c in PRICE_COLS:
        df[c + "_dec"] = df[c].map(to_decimal)
        df[c] = df[c + "_dec"].map(lambda x: float(x) if x is not None else np.nan)
    df["symbol"] = df["instrument_raw_symbol"].astype(str)
    df["ts_utc"] = pd.to_datetime(df["open_ts_ns"], utc=True)
    df["ts_kst"] = df["ts_utc"].dt.tz_convert(KST)
    mode = df["ingest_mode"].astype(str)
    df["is_backfill"] = mode.eq("backfill")
    df["avail_ts_ns"] = np.where(df["is_backfill"], df["close_ts_ns"] + 1_000_000, df["receive_ts_ns"])
    df["trading_date_kst"] = df["ts_kst"].dt.strftime("%Y-%m-%d")
    df["trading_date_mismatch"] = df["trading_date"].astype(str) != df["trading_date_kst"]
    stats["trading_date_mismatch"] = int(df["trading_date_mismatch"].sum())
    df = df.sort_values(["symbol", "open_ts_ns"]).reset_index(drop=True)
    keep = ["symbol", "open_ts_ns", "close_ts_ns", "ts_utc", "ts_kst", "open", "high", "low", "close",
            "volume", "trade_count", "is_backfill", "receive_ts_ns", "avail_ts_ns", "collector_run_id",
            "trading_date", "trading_date_kst", "trading_date_mismatch", "_file"]
    return df[keep], stats


def quality_report(bars: pd.DataFrame, day_start_hour: int = 8) -> dict:
    rep = {}
    for sym, g in bars.groupby("symbol"):
        t = g["ts_kst"]
        block = (t - pd.Timedelta(hours=day_start_hour)).dt.strftime("%Y-%m-%d")
        cov = g.groupby(block).size().rename("bars").to_frame()
        cov["coverage"] = (cov["bars"] / 1440).round(3)
        hours = g.groupby(t.dt.hour).size()
        ndays = max(1, block.nunique())
        gaps = []
        ts = g["open_ts_ns"].to_numpy()
        runs = g["collector_run_id"].astype(str).to_numpy()
        d = np.diff(ts) // 60_000_000_000
        for i in np.where(d > 5)[0]:
            gaps.append({
                "after_kst": str(pd.Timestamp(ts[i], tz="UTC").tz_convert(KST)),
                "before_kst": str(pd.Timestamp(ts[i + 1], tz="UTC").tz_convert(KST)),
                "missing_min": int(d[i] - 1),
                "run_before": runs[i], "run_after": runs[i + 1],
            })
        rep[sym] = {
            "bars": int(len(g)),
            "first_kst": str(t.min()), "last_kst": str(t.max()),
            "backfill_share": round(float(g["is_backfill"].mean()), 4),
            "zero_trade_bars": int((g["trade_count"] == 0).sum()),
            "collector_runs": int(g["collector_run_id"].nunique()),
            "block_coverage": cov.reset_index(names="block").to_dict("records"),
            "bars_by_kst_hour_avg_per_block": {int(h): round(n / ndays, 1) for h, n in hours.items()},
            "gaps_over_5min": gaps,
        }
    return rep


def run_ingest(cfg: dict, raw_dir: str, out_dir: str, report_dir: str) -> tuple[pd.DataFrame, dict, list[str]]:
    df, files = load_raw(raw_dir)
    bars, stats = clean_bars(df)
    os.makedirs(out_dir, exist_ok=True)
    for sym, g in bars.groupby("symbol"):
        g.drop(columns=["_file"]).to_parquet(os.path.join(out_dir, f"bars_1m_{sym}.parquet"), index=False)
    rep = {"clean_stats": stats, "symbols": quality_report(bars, cfg["calendar"]["day_start_hour"]),
           "files": [os.path.basename(f) for f in files]}
    expected = [cfg["symbols"]["local"], cfg["symbols"]["adr"]]
    rep["missing_symbols"] = [s for s in expected if s not in rep["symbols"]]
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, "data_quality.json"), "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1, default=str)
    return bars, rep, files


def load_minute_prices(processed_dir: str, symbols: dict, mid_dir: str | None = None) -> pd.DataFrame:
    """두 심볼의 1분 가격을 open_ts 기준으로 맞춘 표. 없는 심볼은 NaN 열.

    mid_dir 에 mid_1m_<sym>.parquet (src.orderbook) 이 있으면 유효한 분은 중간가, 나머지 분은
    체결가 종가로 대체한다 (3절). <key>_is_mid 열로 표시.
    """
    out = {}
    avail = {}
    for key, sym in symbols.items():
        p = os.path.join(processed_dir, f"bars_1m_{sym}.parquet")
        if os.path.exists(p):
            g = pd.read_parquet(p)
            s = g.set_index("ts_utc")
            price = s["close"].copy()
            av = s["avail_ts_ns"].copy()
            is_mid = pd.Series(False, index=s.index)
            mp = os.path.join(mid_dir, f"mid_1m_{sym}.parquet") if mid_dir else None
            if mp and os.path.exists(mp):
                m = pd.read_parquet(mp)
                m = m[m["valid"]].set_index("ts_utc")
                common = m.index.intersection(price.index)
                extra = m.index.difference(price.index)     # 분봉은 없지만 호가는 있는 분
                price.loc[common] = m.loc[common, "mid"]
                av.loc[common] = m.loc[common, "avail_ts_ns"]
                is_mid.loc[common] = True
                if len(extra):
                    price = pd.concat([price, m.loc[extra, "mid"]]).sort_index()
                    av = pd.concat([av, m.loc[extra, "avail_ts_ns"]]).sort_index()
                    is_mid = pd.concat([is_mid, pd.Series(True, index=extra)]).sort_index()
            out[key] = price
            avail[key] = av
            out[key + "_is_mid"] = is_mid
            out[key + "_backfill"] = s["is_backfill"]
    px = pd.DataFrame(out)
    for k, v in avail.items():
        px[k + "_avail_ts_ns"] = v
    for key in symbols:
        if key not in px:
            px[key] = np.nan
    return px.sort_index()
