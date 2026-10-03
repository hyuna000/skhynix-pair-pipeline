"""3절 오더북: depth@100ms 변경분 -> 분 단위 최우선 호가·중간가 파일.

스냅샷이 없으면 빈 호가창에서 시작한다. 이 경우 호가창은 "실제 호가창의 부분집합"이 된다
(시작 전부터 있던 가격대는 한 번 갱신될 때까지 안 보임). 최우선 호가는 수백 ms 마다 갱신되므로
워밍업 후에는 맞아지고, 이를 체결가 종가와 대조해 검증한다(validate_against_bars).

- 시퀀스 연속성: previous_update_id == 직전 final_update_id. 끊기면 호가창을 비우고 새 구간 시작.
- 분 m 의 값 = 분 m 이 끝나는 시점(다음 분 시작 직전)의 호가창 상태 (분봉 종가와 같은 시점).
- 유효 조건: 구간 시작 후 warmup_sec 경과, 양쪽 호가 존재, bid < ask.
- avail_ts_ns = 그 분 마지막 이벤트의 receive_ts_ns (실제로 알 수 있었던 시각).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

NS_MIN = 60_000_000_000


def _levels(arr):
    if arr is None:
        return []
    return [(float(x["price"]), float(x["quantity"])) for x in arr]


def reconstruct(df: pd.DataFrame, warmup_sec: float = 60.0, snapshot: pd.DataFrame | None = None,
                depth: int = 20) -> pd.DataFrame:
    df = df.drop_duplicates("event_id").sort_values(["final_update_id"]).reset_index(drop=True)
    bids: dict[float, float] = {}
    asks: dict[float, float] = {}
    seg = -1
    seg_start = None
    last_final = None
    rows = []
    cur_min = None
    last_recv = None
    n_ev = 0

    def emit(minute, ts_now):
        valid = (seg_start is not None and (ts_now - seg_start) >= warmup_sec * 1e9 and bids and asks)
        bb = max(bids) if bids else np.nan
        ba = min(asks) if asks else np.nan
        if valid and not bb < ba:
            valid = False
        tb = sorted(bids.items(), key=lambda kv: -kv[0])[:depth] if valid else []
        ta = sorted(asks.items(), key=lambda kv: kv[0])[:depth] if valid else []
        rows.append((minute, bb, ba, last_recv, n_ev, seg, bool(valid),
                     len(bids), len(asks),
                     [p for p, _ in tb], [q for _, q in tb], [p for p, _ in ta], [q for _, q in ta]))

    src = df["source_ts_ns"].to_numpy()
    prev = df["previous_update_id"].to_numpy()
    fin = df["final_update_id"].to_numpy()
    recv = df["receive_ts_ns"].to_numpy()
    B = df["bids"].to_numpy()
    A = df["asks"].to_numpy()
    for i in range(len(df)):
        m = (src[i] // NS_MIN) * NS_MIN
        if cur_min is not None and m != cur_min:
            emit(cur_min, cur_min + NS_MIN - 1)
            # 이벤트 없이 지나간 분은 기록하지 않는다 (재구성 불가 -> 체결가 대체)
            n_ev = 0
        cur_min = m
        if last_final is None or prev[i] != last_final:
            bids.clear(); asks.clear()
            seg += 1
            seg_start = src[i]
        for p, q in _levels(B[i]):
            if q == 0:
                bids.pop(p, None)
            else:
                bids[p] = q
        for p, q in _levels(A[i]):
            if q == 0:
                asks.pop(p, None)
            else:
                asks[p] = q
        last_final = fin[i]
        last_recv = recv[i]
        n_ev += 1
    if cur_min is not None:
        emit(cur_min, cur_min + NS_MIN - 1)
    out = pd.DataFrame(rows, columns=["open_ts_ns", "best_bid", "best_ask", "avail_ts_ns", "n_events",
                                      "segment", "valid", "n_bid_levels", "n_ask_levels",
                                      "bid_px", "bid_qty", "ask_px", "ask_qty"])
    out["mid"] = (out["best_bid"] + out["best_ask"]) / 2
    out["spread_bp"] = (out["best_ask"] - out["best_bid"]) / out["mid"] * 1e4
    out["ts_utc"] = pd.to_datetime(out["open_ts_ns"], utc=True)
    return out


def book_vwap(px: list, qty: list, notional: float) -> tuple[float, bool]:
    """호가 단계를 위에서부터 소진해 notional 달러어치 체결할 때의 평균가. (가격, 깊이 충분 여부)"""
    remain, cost, filled = notional, 0.0, 0.0
    for p, q in zip(px, qty):
        take = min(q, remain / p)
        cost += take * p
        filled += take
        remain -= take * p
        if remain <= 1e-9:
            return cost / filled, True
    return (cost / filled if filled > 0 else float("nan")), False


def build_mid_files(raw_ob_dir: str, out_dir: str, warmup_sec: float = 60.0) -> dict:
    """raw_ob_dir 의 order_book parquet 들을 심볼별로 재구성해 mid_1m_<sym>.parquet 저장."""
    files = sorted(glob.glob(os.path.join(raw_ob_dir, "*.parquet")))
    cols = ["event_id", "event_type", "instrument_raw_symbol", "source_ts_ns", "receive_ts_ns",
            "first_update_id", "final_update_id", "previous_update_id", "update_type", "bids", "asks"]
    by_sym: dict[str, list] = {}
    for f in files:
        d = pd.read_parquet(f, columns=cols)
        d = d[d["event_type"].astype(str) == "order_book"]
        for sym, g in d.groupby(d["instrument_raw_symbol"].astype(str)):
            by_sym.setdefault(sym, []).append(g)
    report = {}
    os.makedirs(out_dir, exist_ok=True)
    for sym, parts in by_sym.items():
        d = pd.concat(parts, ignore_index=True)
        if (d["update_type"].astype(str) != "delta").any():
            report.setdefault("notes", []).append(f"{sym}: delta 외 update_type 존재 (스냅샷 처리 미구현)")
        mid = reconstruct(d, warmup_sec)
        mid.to_parquet(os.path.join(out_dir, f"mid_1m_{sym}.parquet"), index=False)
        report[sym] = {"minutes": int(len(mid)), "valid_minutes": int(mid["valid"].sum()),
                       "segments": int(mid["segment"].max() + 1) if len(mid) else 0,
                       "spread_bp_median": float(mid.loc[mid.valid, "spread_bp"].median()) if mid.valid.any() else None}
    return report


def validate_against_bars(mid: pd.DataFrame, bars: pd.DataFrame) -> pd.DataFrame:
    """분 중간가와 분봉 종가 비교. 종가가 [bid - tick, ask + tick] 근처에 있어야 정상."""
    m = mid[mid["valid"]].set_index("ts_utc")
    b = bars.set_index("ts_utc")["close"]
    j = m.join(b, how="inner")
    j["close_minus_mid_bp"] = (j["close"] - j["mid"]) / j["mid"] * 1e4
    j["close_inside"] = (j["close"] >= j["best_bid"] - 1e-9) & (j["close"] <= j["best_ask"] + 1e-9)
    return j
