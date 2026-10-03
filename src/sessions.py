"""5절 세션: 거래일 [T0, D+1 07:59] 를 펀딩 정산 시각 앞뒤 buffer 초를 피해 나눈다.

분봉 m 은 구간 [m, m+60s) 를 대표한다. 구간이 [정산 - b, 정산 + b] 와 겹치는 분봉은 세션에서 빠진다.
예) 정산 09:00, b=60s -> 08:59, 09:00 분봉 제외 -> 세션은 ..~08:58 분봉(08:59:00 청산), 09:01 분봉부터 재개.
세션 마지막 분봉에서 포지션이 남아 있으면:
  - 거래일 마지막 분봉(07:59)이면 FORCED_0759 (항상 강제 청산)
  - 그 외: policy="exit" 이면 FORCED_FUNDING (계획서 원안),
           policy="hold" 이면 HOLD -> 청산하지 않고 정산을 들고 넘김 (2026-10-03 팀 결정).
           정산 앞뒤 buffer 동안은 진입·청산 주문을 내지 않는다 (목표 도달도 버퍼가 끝난 뒤 처리).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

import pandas as pd

from .funding import FundingSchedule
from .tradecal import TradingCalendar

KST = "Asia/Seoul"


@dataclass
class Session:
    idx: int
    start: pd.Timestamp     # 첫 분봉 open (KST)
    end: pd.Timestamp       # 마지막 분봉 open (KST)
    end_reason: str         # FORCED_FUNDING | FORCED_0759 | HOLD

    @property
    def minutes(self) -> int:
        return int((self.end - self.start) / pd.Timedelta(minutes=1)) + 1


def trading_day_bounds(cal: TradingCalendar, d: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """[T0, 마지막 분봉 open]. 마지막 분봉 = D+1 07:59."""
    t0 = cal.t0(d)
    last = pd.Timestamp(datetime.combine(d + timedelta(days=1), time(cal.h0)), tz=KST) - pd.Timedelta(minutes=1)
    return t0, last


def build_sessions(cal: TradingCalendar, sched: FundingSchedule, d: date, buffer_sec: int = 60,
                   policy: str = "exit") -> list[Session]:
    if policy not in ("exit", "hold"):
        raise ValueError(f"funding policy: {policy}")
    t0, last = trading_day_bounds(cal, d)
    settles = sched.settlement_times(t0 - pd.Timedelta(hours=1), last + pd.Timedelta(hours=1))
    b = pd.Timedelta(seconds=buffer_sec)
    one = pd.Timedelta(minutes=1)
    minutes = pd.date_range(t0, last, freq="1min")
    ok = []
    for m in minutes:
        ms, me = m.tz_convert("UTC"), m.tz_convert("UTC") + one
        bad = any((ms < s + b) and (me > s - b) for s in settles)
        ok.append(not bad)
    sessions, cur = [], None
    for m, good in zip(minutes, ok):
        if good:
            if cur is None:
                cur = [m, m]
            else:
                cur[1] = m
        elif cur is not None:
            sessions.append(cur)
            cur = None
    if cur is not None:
        sessions.append(cur)
    out = []
    for i, (s, e) in enumerate(sessions):
        out.append(Session(i, s, e, "FORCED_0759" if e == last else ("HOLD" if policy == "hold" else "FORCED_FUNDING")))
    return out


def holding_segments(sessions: list[Session]) -> list[int]:
    """포지션이 이어질 수 있는 구간 길이(분). HOLD 로 이어진 세션은 버퍼 분까지 포함해 하나로 합친다.
    임계값 최적화(thresholds.optimize)의 경로 길이로 쓴다."""
    segs, start = [], None
    for ss in sessions:
        if start is None:
            start = ss.start
        if ss.end_reason != "HOLD":
            segs.append(int((ss.end - start) / pd.Timedelta(minutes=1)) + 1)
            start = None
    if start is not None:
        segs.append(int((sessions[-1].end - start) / pd.Timedelta(minutes=1)) + 1)
    return segs
