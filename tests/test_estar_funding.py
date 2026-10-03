import numpy as np
import pandas as pd

from src import estar, thresholds
from src.sessions import Session, segment_layout


def _estar_path(rng, T, phi, g, c=0.0, s=0.3):
    z = np.zeros(T)
    dz_prev = 0.0
    for t in range(1, T):
        dz = phi * z[t - 1] * (1 - np.exp(-g * z[t - 1] ** 2)) + c * dz_prev + s * rng.standard_normal()
        z[t] = z[t - 1] + dz
        dz_prev = dz
    return z


def test_estar_fit_recovers_parameters():
    rng = np.random.default_rng(1)
    z = _estar_path(rng, 6000, -0.3, 0.5, c=0.2)
    m = estar.fit([z], max_lags=4)
    assert m["ok"] and m["p"] >= 1
    assert abs(m["phi"] - (-0.3)) < 0.1
    assert 0.2 < m["gamma"] < 1.5
    assert abs(m["c"][0] - 0.2) < 0.06


def test_estar_fit_rejects_explosive():
    rng = np.random.default_rng(2)
    z = np.cumsum(rng.standard_normal(3000)) * 0.1
    z = z * np.exp(np.linspace(0, 3, 3000))       # 발산하는 경로
    m = estar.fit([z], max_lags=2)
    assert not m["ok"] and m["reason"] in ("ESTAR_NOT_MEAN_REVERTING", "ESTAR_UNSTABLE_RANGE")


def test_estar_simulation_is_stationary_and_matches_window_scale():
    rng = np.random.default_rng(3)
    z = _estar_path(rng, 4000, -0.4, 1.0)
    m = estar.fit([z], max_lags=2)
    Z, bad = estar.simulate(m, rng, 300, 2000)
    assert bad == 0.0
    assert 0.6 < Z[:, 500:].std() / z.std() < 1.5


def test_funding_in_vectorized_rules():
    # 경로: 0분 z=-2 진입(롱), 2분 정산, 4분 z=0 청산
    Z = np.array([[-2.0, -2.0, -2.0, -1.0, 0.0, 0.0]])
    fl = np.zeros(6)
    fl[2] = 0.1
    act = np.array([True, True, False, True, True, True])
    eok = np.array([True, True, False, True, True, False])
    fo = np.zeros(1)
    pnl, ntr = thresholds.rules_vectorized(Z, 1.5, 0.0, 99, 0, 0, 0, True, act, eok, fl, -fl, fo)
    assert ntr[0] == 1 and np.isclose(fo[0], 0.1) and np.isclose(pnl[0], 2.0 - 0.1)
    # 숏 스프레드는 같은 정산에서 반대 부호 (받음)
    pnl2, _ = thresholds.rules_vectorized(-Z, 1.5, 0.0, 99, 0, 0, 0, True, act, eok, fl, -fl)
    assert np.isclose(pnl2[0], 2.0 + 0.1)


def test_funding_z_sign_and_scale():
    cfg = {"trading": {"capital_usd": 10000}}
    syms = {"L": "L", "A": "A"}
    fz = thresholds.funding_z(cfg, 1.0, "struct", 0.002, {"A": 0.0001, "L": 0.0001}, syms)
    # 롱 스프레드: ADR 롱이 0.01% 지급 = 5000*1e-4 = 0.5$ ; z 1단위 = 5000*0.002 = 10$
    assert np.isclose(fz["A"], 0.05) and np.isclose(fz["L"], -0.05)


def test_segment_layout_marks_buffer_and_settlement():
    tz = "Asia/Seoul"
    b = pd.Timestamp("2026-09-30 08:00", tz=tz)
    m = lambda k: b + pd.Timedelta(minutes=k)

    class S:
        def settlements(self, a, e):
            st = pd.Timestamp("2026-09-30 09:00", tz=tz).tz_convert("UTC")
            return [(st, "L")] if a <= st <= e else []

    sess = [Session(0, m(0), m(58), "HOLD"), Session(1, m(61), m(100), "FORCED_0759")]
    lay = segment_layout(sess, S())
    assert len(lay) == 1 and lay[0]["L"] == 101
    assert not lay[0]["act"][59] and not lay[0]["act"][60] and lay[0]["act"][61]
    assert not lay[0]["entry_ok"][58] and lay[0]["entry_ok"][57]
    assert lay[0]["settle"] == [(60, "L")]


def test_block_bootstrap_respects_segments_and_values():
    rng = np.random.default_rng(5)
    segs = [np.arange(0, 50, dtype=float), np.arange(1000, 1030, dtype=float)]
    Z = thresholds.block_bootstrap_paths(rng, segs, 200, 300, 20)
    allowed = set(np.concatenate(segs).tolist())
    assert set(np.unique(Z).tolist()) <= allowed
    d = np.diff(Z, axis=1)
    # 블록 안에서는 +1 씩 진행 (평균 블록 20분 -> 대략 95%), 구간 끝(49, 1029) 다음은 항상 새 블록
    assert (d == 1).mean() > 0.85
    assert not np.any((Z[:, :-1] == 49) & (Z[:, 1:] == 50))


def test_level_shift_lowers_expected_pnl():
    from src.config import load_config
    import copy
    cfg = copy.deepcopy(load_config())
    cfg["thresholds"]["n_paths"] = 300
    params = {"sigma": 0.003, "phi": np.exp(-np.log(2) / 60), "phi_ar1": np.exp(-np.log(2) / 60), "a": 0, "b": 1, "mu": 0}
    syms = {"L": "SKHYNIXUSDT", "A": "SKHYUSDT"}
    r0 = thresholds.optimize(cfg, params, 1.0, "struct", [1439], syms, 1, model="linear", level_sd=0.0)
    r1 = thresholds.optimize(cfg, params, 1.0, "struct", [1439], syms, 1, model="linear", level_sd=1.5)
    assert r1["exp_pnl_usd"] < r0["exp_pnl_usd"]
