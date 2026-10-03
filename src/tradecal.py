"""4·5·6절 달력: 정상 블록 판정, 거래일 T0, 윈도우(V1/V2, 주말·휴일 A안), 레짐 라벨.

블록 S = [S 08:00, S+1 08:00) KST.  정상 블록 = S 가 평일이고 한국·미국 모두 휴장이 아닌 날.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta

import pandas as pd

KST = "Asia/Seoul"


@dataclass
class Window:
    trading_date: date
    version: str               # "V1" | "V2"
    t0: pd.Timestamp           # 결정 시각 (KST)
    start: pd.Timestamp        # [start, end)  KST
    end: pd.Timestamp
    day_type: str              # 정상 | 주말A안 | 공휴일 | 휴일직후
    blocks: list[str]
    flags: list[str] = field(default_factory=list)

    @property
    def expected_minutes(self) -> int:
        return int((self.end - self.start) / pd.Timedelta(minutes=1))

    @property
    def key(self) -> tuple:
        return (self.start, self.end)


class TradingCalendar:
    def __init__(self, cfg: dict):
        c = cfg["calendar"]
        self.h0 = c["day_start_hour"]
        self.hsat = c["saturday_start_hour"]
        self.kr = {date.fromisoformat(d) for d in c["kr_holidays"]}
        self.us = {date.fromisoformat(d) for d in c["us_holidays"]}
        # "blocks" = 계획서 6절 (정상 블록만, 주말 A안) | "continuous" = T0 직전 연속 N분 (주말·공휴일 포함)
        self.mode = c.get("window_mode", "blocks")
        self.minutes = {"V1": int(c.get("v1_minutes", 1440)), "V2": int(c.get("v2_minutes", 2880))}

    # ---- 기본 판정 ---------------------------------------------------------
    def is_normal_block(self, d: date) -> bool:
        return d.weekday() < 5 and d not in self.kr and d not in self.us

    def ts(self, d: date, hour: int) -> pd.Timestamp:
        return pd.Timestamp(datetime.combine(d, time(hour)), tz=KST)

    def t0(self, d: date) -> pd.Timestamp:
        return self.ts(d, self.hsat if d.weekday() == 5 else self.h0)

    def block_range(self, s: date) -> tuple[pd.Timestamp, pd.Timestamp]:
        return self.ts(s, self.h0), self.ts(s + timedelta(days=1), self.h0)

    def last_normal_block(self, before: date, max_back: int = 14) -> date | None:
        d = before
        for _ in range(max_back):
            if self.is_normal_block(d):
                return d
            d -= timedelta(days=1)
        return None

    def day_type(self, d: date) -> str:
        wd = d.weekday()
        if wd in (5, 6) or wd == 0:
            return "주말A안"
        if d in self.kr or d in self.us:
            return "공휴일"
        if not self.is_normal_block(d - timedelta(days=1)):
            return "휴일직후"
        return "정상"

    # ---- 윈도우 ------------------------------------------------------------
    def _saturday_of(self, d: date) -> date:
        """일·월 거래일이 재사용할 토요일."""
        wd = d.weekday()
        return d - timedelta(days={5: 0, 6: 1, 0: 2}[wd])

    def window(self, d: date, version: str) -> Window:
        if self.mode == "continuous":
            return self.window_continuous(d, version)
        flags: list[str] = []
        dtype = self.day_type(d)
        t0 = self.t0(d)
        wd = d.weekday()

        if wd in (5, 6, 0):                                   # 토·일·월: 토 09:00 고정 윈도우
            sat = self._saturday_of(d)
            if sat != d:
                flags.append(f"REUSE_SAT_{sat.isoformat()}")
            fri = sat - timedelta(days=1)
            end = self.ts(sat, self.hsat)                     # 토 09:00
            if self.is_normal_block(fri):
                last = fri
            else:
                last = self.last_normal_block(fri)
                flags.append("FRI_NOT_NORMAL")
            if last is None:
                return Window(d, version, t0, end, end, dtype, [], flags + ["NO_NORMAL_BLOCK"])
            blocks = [last]
            if version == "V2":
                prev = last - timedelta(days=1)
                if self.is_normal_block(prev):
                    blocks = [prev, last]
                else:
                    flags.append("V2_FALLBACK_V1")
            start = self.block_range(blocks[0])[0]
            if last != fri:                                    # 금요일이 비정상이면 연장 없이 그 블록 끝까지
                end = self.block_range(last)[1]
                flags.append("NO_SAT_EXTENSION")
            return Window(d, version, t0, start, end, dtype, [b.isoformat() for b in blocks], flags)

        # 화~금 (공휴일 포함): 직전 정상 블록
        prev = d - timedelta(days=1)
        last = self.last_normal_block(prev)
        if last is None:
            return Window(d, version, t0, t0, t0, dtype, [], ["NO_NORMAL_BLOCK"])
        if last != prev:
            flags.append(f"STALE_BLOCK_{last.isoformat()}")
        blocks = [last]
        if version == "V2":
            p2 = last - timedelta(days=1)
            if self.is_normal_block(p2):
                blocks = [p2, last]
            else:
                flags.append("V2_FALLBACK_V1")
        start = self.block_range(blocks[0])[0]
        end = self.block_range(last)[1]
        return Window(d, version, t0, start, end, dtype, [b.isoformat() for b in blocks], flags)

    def window_continuous(self, d: date, version: str) -> Window:
        """T0 직전 연속 N분. 블록 정상 여부를 따지지 않고 주말·공휴일 데이터도 쓴다. 날마다 새로 추정."""
        t0 = self.t0(d)
        n = self.minutes[version]
        start = t0 - pd.Timedelta(minutes=n)
        blocks = sorted({(start + pd.Timedelta(minutes=m) - pd.Timedelta(hours=self.h0)).date().isoformat()
                         for m in range(0, n, 60)})
        abnormal = [b for b in blocks if not self.is_normal_block(date.fromisoformat(b))]
        flags = ["CONTINUOUS"] + ([f"INCLUDES_NONNORMAL_{'+'.join(abnormal)}"] if abnormal else [])
        return Window(d, version, t0, start, t0, self.day_type(d), blocks, flags)

    # ---- 4절 레짐 (거래 규칙에는 쓰지 않음) ----------------------------------
    def regime(self, ts: pd.Timestamp, k_start: int = 8, k_end: int = 20) -> str:
        ts = ts.tz_convert(KST)
        d = ts.date()
        kr_open = d.weekday() < 5 and d not in self.kr and k_start <= ts.hour < k_end
        # 미국 갱신: 월 09:00 ~ 토 09:00 KST, 미국 휴일(미국 날짜 기준)은 해당 KST 구간 제외
        us_date = (ts - pd.Timedelta(hours=13)).date()   # EDT 기준 미국 날짜 근사
        wd = ts.weekday()
        in_week = not ((wd == 5 and ts.hour >= 9) or wd == 6 or (wd == 0 and ts.hour < 9))
        us_open = in_week and us_date not in self.us
        if kr_open and us_open:
            return "K"
        if us_open:
            return "U"
        if kr_open:
            return "KO"
        return "W"
