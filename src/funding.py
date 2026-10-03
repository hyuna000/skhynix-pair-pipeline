"""10절 펀딩: 원자료 -> 정산 이력 + 정산 시각표.

정산 시각표는 데이터에 실제로 찍힌 정산 시각에서 주기를 추정한다 (funding_interval_hours 필드는
2026-09 데이터에서 실제 주기(4시간)와 달리 8 로 남아 있어 신뢰하지 않는다).
데이터가 없는 심볼은 config 의 기본 주기·기준 시각을 쓰고 ASSUMED 로 표시한다.
두 레그 중 하나라도 정산하는 시각은 모두 세션 경계가 된다 (합집합).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

KST = "Asia/Seoul"
NS_H = 3_600_000_000_000


def load_funding(raw_dir: str) -> pd.DataFrame:
    files = sorted(glob.glob(os.path.join(raw_dir, "*.parquet")))
    if not files:
        return pd.DataFrame(columns=["symbol", "settle_ts", "rate", "interval_field", "avail_ts_ns"])
    d = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    d = d[d["event_type"].astype(str) == "funding_rate"].drop_duplicates("event_id")
    out = pd.DataFrame({
        "symbol": d["instrument_raw_symbol"].astype(str),
        # 정산 시각은 정시 + 수 ms 로 찍힌다 -> 초 단위로 내림
        "settle_ts": pd.to_datetime(d["source_ts_ns"], utc=True).dt.floor("s"),
        "rate": d["funding_rate"].map(lambda x: float(x) if x is not None else np.nan),
        "interval_field": d["funding_interval_hours"],
        "avail_ts_ns": d["receive_ts_ns"],
    })
    return out.sort_values(["symbol", "settle_ts"]).drop_duplicates(["symbol", "settle_ts"]).reset_index(drop=True)


def infer_interval_hours(times: pd.Series) -> int | None:
    """연속 정산 시각 간격의 최소 공통 단위(시간). 데이터에 빈칸이 많아도 최솟값이 실제 주기."""
    t = times.sort_values().drop_duplicates()
    if len(t) < 2:
        return None
    gaps = (np.diff(t.astype("int64").to_numpy()) / NS_H).round().astype(int)
    gaps = gaps[gaps > 0]
    if len(gaps) == 0:
        return None
    return int(np.gcd.reduce(gaps))


class FundingSchedule:
    def __init__(self, cfg: dict, funding: pd.DataFrame):
        fc = cfg["funding"]
        self.symbols = [cfg["symbols"]["local"], cfg["symbols"]["adr"]]
        self.info = {}
        for sym in self.symbols:
            t = funding.loc[funding["symbol"] == sym, "settle_ts"]
            h = infer_interval_hours(t)
            if h:
                anchor = t.min()
                field = funding.loc[funding["symbol"] == sym, "interval_field"]
                self.info[sym] = {"interval_h": h, "anchor": anchor, "source": "data",
                                  "n_settlements": int(len(t)),
                                  "interval_field_values": sorted(set(int(x) for x in field.dropna()))}
            else:
                anchor = pd.Timestamp(fc["default_anchor_kst"], tz=KST).tz_convert("UTC")
                self.info[sym] = {"interval_h": int(fc["default_interval_hours"]), "anchor": anchor,
                                  "source": "ASSUMED", "n_settlements": 0, "interval_field_values": []}
        self.funding = funding

    def settlements(self, start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, str]]:
        """[start, end] 안의 (정산시각 UTC, 심볼) 목록. 시각표 기준 (데이터 누락과 무관)."""
        out = []
        s, e = start.tz_convert("UTC"), end.tz_convert("UTC")
        for sym, inf in self.info.items():
            step = pd.Timedelta(hours=inf["interval_h"])
            k0 = int(np.floor((s - inf["anchor"]) / step))
            t = inf["anchor"] + k0 * step
            while t <= e:
                if t >= s:
                    out.append((t, sym))
                t += step
        return sorted(out)

    def settlement_times(self, start, end) -> list[pd.Timestamp]:
        return sorted({t for t, _ in self.settlements(start, end)})

    def rate(self, sym: str, t: pd.Timestamp) -> float:
        f = self.funding
        m = f[(f["symbol"] == sym) & (f["settle_ts"] == t.tz_convert("UTC").floor("s"))]
        return float(m["rate"].iloc[0]) if len(m) else np.nan

    def summary(self) -> dict:
        return {s: {k: (str(v) if isinstance(v, pd.Timestamp) else v) for k, v in i.items()} for s, i in self.info.items()}
