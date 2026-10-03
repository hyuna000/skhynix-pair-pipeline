from datetime import date
from src.config import load_config
from src.tradecal import TradingCalendar

_cfg = load_config()
_cfg["calendar"]["window_mode"] = "blocks"   # 계획서 6절 규칙 검증 (기본 설정은 continuous)
cal = TradingCalendar(_cfg)

def w(d, v):
    x = cal.window(date.fromisoformat(d), v)
    return x.start.strftime("%m-%d %H:%M"), x.end.strftime("%m-%d %H:%M"), x.expected_minutes, x.day_type, x.flags

def test_weekday():
    assert w("2026-09-30", "V1")[:3] == ("09-29 08:00", "09-30 08:00", 1440)
    assert w("2026-09-30", "V2")[:3] == ("09-28 08:00", "09-30 08:00", 2880)

def test_saturday_extension():
    assert w("2026-10-03", "V1")[:3] == ("10-02 08:00", "10-03 09:00", 1500)
    assert w("2026-10-03", "V2")[:3] == ("10-01 08:00", "10-03 09:00", 2940)

def test_sun_mon_reuse_sat():
    assert w("2026-10-04", "V1")[:2] == w("2026-10-03", "V1")[:2]
    # 10-05 월 (한국 대체공휴일) 도 토요일 윈도우
    assert w("2026-10-05", "V1")[:2] == w("2026-10-03", "V1")[:2]

def test_after_holiday():
    # 10-06 화: 10-05 블록 비정상 -> 최근 정상 블록 10-02 (금), 연장 없음
    s, e, n, dt, fl = w("2026-10-06", "V1")
    assert (s, e, n, dt) == ("10-02 08:00", "10-03 08:00", 1440, "휴일직후")
    # V2: 10-01, 10-02 연속 정상
    assert w("2026-10-06", "V2")[:3] == ("10-01 08:00", "10-03 08:00", 2880)

def test_chuseok():
    # 09-24(목) 추석 당일: 09-23 블록 정상
    assert w("2026-09-24", "V1")[:4] == ("09-23 08:00", "09-24 08:00", 1440, "공휴일")
    # 09-25(금): 09-24 블록 비정상 -> 09-23 블록 재사용
    assert w("2026-09-25", "V1")[:2] == ("09-23 08:00", "09-24 08:00")
    # 09-26(토): 금요일(09-25) 휴장 -> 09-23 블록, 연장 없음
    s, e, n, dt, fl = w("2026-09-26", "V1")
    assert (s, e) == ("09-23 08:00", "09-24 08:00") and "FRI_NOT_NORMAL" in fl
    # V2 for 09-25: 09-22, 09-23 연속 정상
    assert w("2026-09-25", "V2")[:2] == ("09-22 08:00", "09-24 08:00")

def test_v2_fallback():
    # 09-29(화): V2 는 09-27(일)+09-28 -> 일요일 비정상 -> V1 대체
    s, e, n, dt, fl = w("2026-09-29", "V2")
    assert n == 1440 and "V2_FALLBACK_V1" in fl

def test_regime():
    import pandas as pd
    r = lambda s: cal.regime(pd.Timestamp(s, tz="Asia/Seoul"))
    assert r("2026-09-30 10:00") == "K"
    assert r("2026-09-30 22:00") == "U"
    assert r("2026-09-28 08:30") == "KO"
    assert r("2026-09-27 12:00") == "W"
    assert r("2026-09-24 10:00") == "U"   # 추석: 한국 휴장, 미국 개장

def test_continuous_mode():
    import copy
    cfg = load_config(); cfg = copy.deepcopy(cfg); cfg["calendar"]["window_mode"] = "continuous"
    c = TradingCalendar(cfg)
    w1 = c.window(date.fromisoformat("2026-09-28"), "V2")   # 월요일: 토·일 포함 연속 2880분
    assert (w1.start.strftime("%m-%d %H:%M"), w1.end.strftime("%m-%d %H:%M"), w1.expected_minutes) == ("09-26 08:00", "09-28 08:00", 2880)
    assert any(f.startswith("INCLUDES_NONNORMAL") for f in w1.flags)
    w2 = c.window(date.fromisoformat("2026-10-03"), "V1")   # 토요일 T0 09:00
    assert (w2.start.strftime("%m-%d %H:%M"), w2.end.strftime("%m-%d %H:%M")) == ("10-02 09:00", "10-03 09:00")
