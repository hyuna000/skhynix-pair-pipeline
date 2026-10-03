"""ADR 데이터가 오기 전 파이프라인 작동 확인용 합성 쌍. 실제 데이터 아님.

블록(08:00~07:59)마다 스프레드 성질을 정해 둔다 (정답표 data/synthetic/truth.csv):
  MR : OU 스프레드, 반감기 HL 분 (평균 0, sd 0.003)  -> 게이트 통과가 기대됨
  RW : 랜덤워크 스프레드                              -> 게이트 미통과가 기대됨
가격: ln P_본주 = 효율가격(GARCH형 변동성 + 일중 패턴) + 호가 바운스
      ln P_ADR  = ln P_본주 + ln 0.1 + 0.02(프리미엄) + 스프레드 + 바운스
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "synthetic")

START = pd.Timestamp("2026-09-14 08:00", tz="Asia/Seoul")
END = pd.Timestamp("2026-10-03 09:00", tz="Asia/Seoul")
RW_BLOCKS = {"2026-09-17", "2026-09-22", "2026-09-29", "2026-09-30"}
HL_BY_BLOCK = {"2026-09-15": 20, "2026-09-25": 240}   # 그 외 MR 블록은 45분


def main(seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(START, END, freq="1min", inclusive="left")
    n = len(idx)
    block = (idx - pd.Timedelta(hours=8)).strftime("%Y-%m-%d")
    hour = idx.hour
    # 효율 가격: GARCH(1,1)형 변동성 x 일중 패턴
    base = 0.0006
    intraday = np.where((hour >= 8) & (hour < 20), 1.4, 0.8)
    h = np.empty(n)
    r = np.empty(n)
    h[0] = 1.0
    z = rng.standard_normal(n)
    for t in range(n):
        if t:
            h[t] = 0.05 + 0.10 * (r[t - 1] ** 2) + 0.85 * h[t - 1]
        r[t] = np.sqrt(h[t]) * z[t]
    m = np.log(1300.0) + np.cumsum(base * intraday * r)
    # 스프레드
    s = np.zeros(n)
    truth = []
    for blk in pd.unique(block):
        sel = np.where(block == blk)[0]
        if blk in RW_BLOCKS:
            kind, hl = "RW", np.nan
        else:
            kind, hl = "MR", HL_BY_BLOCK.get(blk, 45)
        truth.append({"block": blk, "regime": kind, "half_life_min": hl})
        prev = s[sel[0] - 1] if sel[0] > 0 else 0.0
        if kind == "RW":
            steps = 0.0003 * rng.standard_normal(len(sel))
            s[sel] = prev + np.cumsum(steps)
        else:
            phi = np.exp(-np.log(2) / hl)
            sd = 0.003 * np.sqrt(1 - phi * phi)
            x = prev * 0.0
            for i, t in enumerate(sel):
                x = phi * x + sd * rng.standard_normal()
                s[t] = x
    lnL = m + 0.0002 * rng.standard_normal(n)
    lnA = m + np.log(0.1) + 0.02 + s + 0.0003 * rng.standard_normal(n)

    proc = os.path.join(OUT, "processed")
    os.makedirs(proc, exist_ok=True)
    ts_utc = idx.tz_convert("UTC")
    for sym, lp in (("SKHYNIXUSDT", lnL), ("SKHYUSDT", lnA)):
        close_ns = ts_utc.as_unit("ns").asi8 + 59_999_000_000
        df = pd.DataFrame({"ts_utc": ts_utc, "close": np.exp(lp), "is_backfill": False,
                           "avail_ts_ns": close_ns + 2_000_000_000})
        df.to_parquet(os.path.join(proc, f"bars_1m_{sym}.parquet"), index=False)
    pd.DataFrame(truth).to_csv(os.path.join(OUT, "truth.csv"), index=False)
    print(f"{n} 분, 블록 {len(truth)}개 -> {proc}")


if __name__ == "__main__":
    main()
