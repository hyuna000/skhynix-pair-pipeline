import copy

import numpy as np
import pandas as pd

from src import engine, thresholds
from src.config import load_config
from src.sessions import Session

SYMS = {"L": "SKHYNIXUSDT", "A": "SKHYUSDT"}


class NoFunding:
    def settlements(self, a, b):
        return []


def _cfg(zero_cost=True):
    cfg = copy.deepcopy(load_config())
    if zero_cost:
        cfg["costs"]["taker_fee"] = 0.0
        cfg["costs"]["half_spread_bp"] = {}
        cfg["costs"]["extra_slippage_bp"] = 0.0
    return cfg


def _market_from_z(z, sigma=0.003, mu=0.0, a=-2.28, L=1300.0, start="2026-09-30 00:00"):
    idx = pd.date_range(start, periods=len(z), freq="1min", tz="UTC")
    lnA = a + np.log(L) + mu + sigma * np.asarray(z)
    px = pd.DataFrame({"L": L, "A": np.exp(lnA), "L_is_mid": False, "A_is_mid": False}, index=idx)
    closes = px[["L", "A"]]
    opens = closes.shift(1)          # 다음 분 시가 = 이번 분 종가 -> 슬리피지 0 이면 신호가격에 체결
    return engine.MarketDay(px, opens, closes, {}, SYMS), idx


def test_engine_matches_vectorized_rules():
    cfg = _cfg()
    rng = np.random.default_rng(3)
    for trial in range(5):
        z = thresholds.simulate_paths(rng, 1, 400, np.exp(-np.log(2) / 30), 0.9)[0]
        md, idx = _market_from_z(z)
        sess = [Session(0, idx[0].tz_convert("Asia/Seoul"), idx[-2].tz_convert("Asia/Seoul"), "FORCED_FUNDING")]
        e, x, s = 1.25, 0.0, 3.0
        thr = {"entry_z": e, "exit_z": x, "stop_z": s}
        params = {"a": -2.28, "b": 1.0, "mu": 0.0, "sigma": 0.003}
        res = engine.run_day(cfg, md, sess, params, thr, "struct", NoFunding(), list(idx[:-1]), "2026-09-30")
        pnl_v, ntr_v = thresholds.rules_vectorized(z[None, :-1], e, x, s, 0, 0, 0, rearm=True)
        tr = res["trades"]
        assert len(tr) == int(ntr_v[0])
        zsum = sum((1 if t["direction"] == "LONG_SPREAD" else -1) * (t["exit_z"] - t["entry_z"]) for t in tr)
        assert np.isclose(zsum, pnl_v[0])


def test_hand_checked_pnl():
    cfg = _cfg(zero_cost=False)
    cfg["costs"]["half_spread_bp"] = {}
    cfg["costs"]["extra_slippage_bp"] = 0.0     # 수수료만
    # z: 0 -> -2 (롱 스프레드 진입) -> 0 (TARGET 청산)
    z = np.array([0.0, -2.0, -1.0, 0.0, 0.0, 0.0])
    md, idx = _market_from_z(z)
    sess = [Session(0, idx[0].tz_convert("Asia/Seoul"), idx[-2].tz_convert("Asia/Seoul"), "FORCED_FUNDING")]
    thr = {"entry_z": 1.5, "exit_z": 0.0, "stop_z": 99.0}
    params = {"a": -2.28, "b": 1.0, "mu": 0.0, "sigma": 0.003}
    res = engine.run_day(cfg, md, sess, params, thr, "struct", NoFunding(), list(idx[:-1]), "2026-09-30")
    t = res["trades"][0]
    assert t["direction"] == "LONG_SPREAD" and t["exit_type"] == "TARGET"
    A0, A1 = md.px["A"].iloc[1], md.px["A"].iloc[3]
    qA = np.floor(5000 / A0 / 0.001) * 0.001
    qL = np.floor(5000 / 1300.0 / 0.001) * 0.001
    gross = qA * (A1 - A0)                   # 본주 가격은 고정 -> 본주 레그 손익 0
    fees = 0.0005 * (qA * (A0 + A1) + qL * 2 * 1300.0)
    assert np.isclose(t["gross_pnl"], gross)
    assert np.isclose(t["fees"], fees)
    assert np.isclose(t["net_pnl"], gross - fees)
    # 손익 근사: C/2 * Δs = 5000 * 2 * sigma
    assert abs(gross - 5000 * 2 * 0.003) / (5000 * 2 * 0.003) < 0.01


def test_forced_exit_at_session_end_and_no_entry_on_last_minute():
    cfg = _cfg()
    z = np.array([0.0, -2.0, -2.0, -2.0, -2.5, -2.5])
    md, idx = _market_from_z(z)
    tz = "Asia/Seoul"
    sess = [Session(0, idx[0].tz_convert(tz), idx[2].tz_convert(tz), "FORCED_FUNDING"),
            Session(1, idx[4].tz_convert(tz), idx[4].tz_convert(tz), "FORCED_0759")]
    thr = {"entry_z": 1.5, "exit_z": 0.0, "stop_z": 99.0}
    params = {"a": -2.28, "b": 1.0, "mu": 0.0, "sigma": 0.003}
    res = engine.run_day(cfg, md, sess, params, thr, "struct", NoFunding(), list(idx[:-1]), "2026-09-30")
    assert [t["exit_type"] for t in res["trades"]] == ["FORCED_FUNDING"]
    reasons = {s["minute_utc"]: s["skip_reason"] for s in res["signals"]}
    assert reasons[str(idx[3])] == "FUNDING_BLACKOUT"       # 세션 사이
    assert reasons[str(idx[4])] == "AFTER_CUTOFF"           # 1분짜리 세션의 마지막 분


def test_stop_then_rearm():
    cfg = _cfg()
    z = np.array([0.0, 2.0, 3.5, 3.0, 0.5, 2.0, -0.2, 0.0])
    md, idx = _market_from_z(z)
    sess = [Session(0, idx[0].tz_convert("Asia/Seoul"), idx[-2].tz_convert("Asia/Seoul"), "FORCED_FUNDING")]
    thr = {"entry_z": 1.5, "exit_z": 0.0, "stop_z": 3.0}
    params = {"a": -2.28, "b": 1.0, "mu": 0.0, "sigma": 0.003}
    res = engine.run_day(cfg, md, sess, params, thr, "struct", NoFunding(), list(idx[:-1]), "2026-09-30")
    assert [t["exit_type"] for t in res["trades"]] == ["STOP", "TARGET"]
    assert any(s["skip_reason"] == "NOT_ARMED" for s in res["signals"])   # z=3.0 은 손절 직후라 재진입 안 함


class OneFunding:
    """idx[3] + 30초에 ADR 정산 1회."""
    def __init__(self, t, rate):
        self.t, self.r = t, rate

    def settlements(self, a, b):
        return [(self.t, SYMS["A"])] if a <= self.t <= b else []

    def rate(self, sym, t):
        return self.r


def test_hold_through_funding_and_defer_exit_in_buffer():
    cfg = _cfg()
    z = np.array([0.0, -2.0, -2.0, 0.0, -1.0, 0.0, 0.0, 0.0])
    md, idx = _market_from_z(z)
    tz = "Asia/Seoul"
    sess = [Session(0, idx[0].tz_convert(tz), idx[2].tz_convert(tz), "HOLD"),
            Session(1, idx[4].tz_convert(tz), idx[-2].tz_convert(tz), "FORCED_0759")]
    thr = {"entry_z": 1.5, "exit_z": 0.0, "stop_z": 99.0}
    params = {"a": -2.28, "b": 1.0, "mu": 0.0, "sigma": 0.003}
    st = idx[3] + pd.Timedelta(seconds=30)
    for rate, n_miss in ((0.0001, 0), (np.nan, 1)):
        res = engine.run_day(cfg, md, sess, params, thr, "struct", OneFunding(st, rate), list(idx[:-1]), "2026-09-30")
        t = res["trades"]
        assert len(t) == 1 and t[0]["exit_type"] == "TARGET"
        assert t[0]["exit_signal_utc"] == str(idx[5])          # idx[3] 의 z=0 은 버퍼라 무시, idx[4] 는 z=-1
        assert t[0]["funding_events"] == 1 and t[0]["funding_missing"] == n_miss
        f = res["funding"][0]
        if n_miss:
            assert f["rate_source"] == "MISSING_ASSUMED" and f["amount"] == 0.0
        else:
            # 롱 스프레드 = ADR 롱 -> 양수 펀딩률이면 지급
            assert f["amount"] < 0 and np.isclose(t[0]["funding"], -f["amount"])
            assert np.isclose(t[0]["net_pnl"], t[0]["gross_pnl"] - t[0]["fees"] - t[0]["funding"])


def test_holding_segments():
    from src.sessions import holding_segments
    tz = "Asia/Seoul"
    base = pd.Timestamp("2026-09-30 08:00", tz=tz)
    m = lambda k: base + pd.Timedelta(minutes=k)
    sess = [Session(0, m(0), m(58), "HOLD"), Session(1, m(61), m(298), "HOLD"), Session(2, m(301), m(400), "FORCED_0759")]
    assert holding_segments(sess) == [401]
    sess2 = [Session(0, m(0), m(58), "FORCED_FUNDING"), Session(1, m(61), m(400), "FORCED_0759")]
    assert holding_segments(sess2) == [59, 340]
