"""9절 임계값: 세션 길이를 반영한 시뮬레이션으로 진입·청산·손절을 고른다 (Bertram 값은 참고 기록).

z 단위 모형 (관측 스프레드의 윈도우 표준편차 = 1):
  잠재 AR(1)  x_t = phi x_{t-1} + eta,  Var(x) = sx2
  관측        z_t = x_t + e_t,          Var(e) = 1 - sx2
  sx2 = phi_ar1 / phi_arma  (AR(1)+백색잡음이면 관측 1차 자기상관 = phi * sx2)

매매 규칙 (engine.py 와 같은 규칙, rules_vectorized 를 테스트로 대조):
  매 분 순서: (1) 청산 판정 -> (2) 재무장 -> (3) 진입 판정 (세션 마지막 분은 진입 없음)
  롱 스프레드(z < -entry 진입): z >= -exit 이면 TARGET, z <= -stop 이면 STOP
  숏 스프레드(z > entry 진입) : z <=  exit 이면 TARGET, z >=  stop 이면 STOP
  세션 마지막 분에 남은 포지션은 강제 청산.  손절 뒤에는 |z| < entry 가 될 때까지 재진입 금지.

비용 (z 단위): 왕복 비용 c_z = (1+b) * sum_leg w_leg * 2*(fee + slip_leg) / sigma
  (손익 = C/(1+b) * Δs 이므로 달러 비용을 C/(1+b) 와 sigma 로 나눈 값). 강제 청산은 청산쪽 슬리피지 x mult.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.special import erfi


def leg_weights(b: float, candidate: str) -> tuple[float, float]:
    """(w_A, w_L). 후보 4 는 50:50."""
    if candidate == "struct":
        return 0.5, 0.5
    return 1.0 / (1.0 + b), b / (1.0 + b)


def cost_z(cfg: dict, b: float, candidate: str, sigma: float, symbols: dict) -> dict:
    c = cfg["costs"]
    fee = c["taker_fee"]
    hs = c["half_spread_bp"]
    slip = {k: (hs.get(sym, 0.0) + c["extra_slippage_bp"]) * 1e-4 for k, sym in symbols.items()}
    wA, wL = leg_weights(b, candidate)
    scale = (1.0 + (1.0 if candidate == "struct" else b)) / sigma
    entry = scale * (wA * (fee + slip["A"]) + wL * (fee + slip["L"]))
    exit_ = entry
    m = c["forced_exit_slippage_mult"]
    forced = scale * (wA * (fee + m * slip["A"]) + wL * (fee + m * slip["L"]))
    return {"entry": entry, "exit": exit_, "forced": forced}


def rules_vectorized(Z, entry, exit_, stop, c_entry, c_exit, c_forced, rearm=True):
    """Z (N paths, L minutes). 반환: 경로별 순손익(z 단위), 거래 수."""
    N, L = Z.shape
    pos = np.zeros(N, np.int8)
    ez = np.zeros(N)
    pnl = np.zeros(N)
    ntr = np.zeros(N, np.int32)
    armed = np.ones(N, bool)
    for t in range(L):
        zt = Z[:, t]
        last = t == L - 1
        if pos.any():
            lg, sh = pos == 1, pos == -1
            tgt = (lg & (zt >= -exit_)) | (sh & (zt <= exit_))
            stp = ((lg & (zt <= -stop)) | (sh & (zt >= stop))) & ~tgt
            fin = (pos != 0) & last & ~tgt & ~stp
            ex = tgt | stp | fin
            pnl[ex] += pos[ex] * (zt[ex] - ez[ex])
            pnl[tgt | stp] -= c_exit
            pnl[fin] -= c_forced
            if rearm:
                armed[stp] = False
            pos[ex] = 0
        armed |= np.abs(zt) < entry
        if not last:
            can = (pos == 0) & armed
            enl = can & (zt < -entry)
            ens = can & (zt > entry)
            new = enl | ens
            pos[enl] = 1
            pos[ens] = -1
            ez[new] = zt[new]
            pnl[new] -= c_entry
            ntr[new] += 1
    return pnl, ntr


def simulate_paths(rng, n, L, phi, sx2):
    sx2 = float(np.clip(sx2, 1e-6, 1.0))
    eta = np.sqrt(sx2 * (1 - phi * phi))
    x = np.empty((n, L))
    x[:, 0] = rng.standard_normal(n) * np.sqrt(sx2)
    shocks = rng.standard_normal((n, L)) * eta
    for t in range(1, L):
        x[:, t] = phi * x[:, t - 1] + shocks[:, t]
    return x + rng.standard_normal((n, L)) * np.sqrt(1.0 - sx2)


def bertram_entry(theta: float, c_latent: float) -> float:
    """Bertram(2010): 진입 a(<0), 청산 m=-a, 단위시간 기대수익 최대. 잠재 z 단위 |a| 를 돌려준다."""
    if not (theta > 0) or not np.isfinite(c_latent):
        return np.nan
    a = np.linspace(0.05, 4.0, 400)
    et = (2 * np.pi / theta) * erfi(a / np.sqrt(2))
    mu = (2 * a - c_latent) / et
    return float(a[np.argmax(mu)])


def optimize(cfg, params: dict, b: float, candidate: str, session_lengths: list[int], symbols: dict, seed: int):
    """그 거래일의 세션 길이 목록에 대해 하루 기대 순수익이 최대인 (entry, exit, stop)."""
    tc = cfg["thresholds"]
    sigma = params["sigma"]
    phi = params["phi"]
    phi1 = params.get("phi_ar1", phi)
    if not (0 < phi < 1):
        phi = phi1 if 0 < phi1 < 1 else np.nan
    if not np.isfinite(phi) or not (sigma > 0):
        return {"ok": False, "reason": "BAD_PARAMS"}
    sx2 = float(np.clip(phi1 / phi, 0.05, 1.0)) if 0 < phi1 < 1 else 1.0
    cz = cost_z(cfg, b, candidate, sigma, symbols)
    rng = np.random.default_rng(seed)
    n = int(tc["n_paths"])
    lengths = {}
    for L in session_lengths:
        lengths[L] = lengths.get(L, 0) + 1
    paths = {L: simulate_paths(rng, n, L, phi, sx2) for L in lengths}
    rearm = bool(cfg["trading"]["rearm_after_stop"])
    best = None
    for e, x, s in itertools.product(tc["entry_grid"], tc["exit_grid"], tc["stop_grid"]):
        if s <= e or x >= e:
            continue
        tot_z, tot_tr = 0.0, 0.0
        for L, cnt in lengths.items():
            p, ntr = rules_vectorized(paths[L], e, x, s, cz["entry"], cz["exit"], cz["forced"], rearm)
            tot_z += cnt * p.mean()
            tot_tr += cnt * ntr.mean()
        if best is None or tot_z > best[0]:
            best = (tot_z, e, x, s, tot_tr)
    tot_z, e, x, s, ntr = best
    capital = cfg["trading"]["capital_usd"]
    unit = capital / (1.0 + (1.0 if candidate == "struct" else b)) * sigma   # z 1 단위의 달러 가치
    theta = -np.log(phi)
    bz = bertram_entry(theta, cz["entry"] * 2 / np.sqrt(sx2))
    return {"ok": True, "entry_z": e, "exit_z": x, "stop_z": s,
            "exp_pnl_usd": float(tot_z * unit), "exp_trades": float(ntr),
            "cost_roundtrip_z": float(cz["entry"] + cz["exit"]), "latent_var_share": sx2,
            "bertram_entry_z": float(bz * np.sqrt(sx2)) if np.isfinite(bz) else np.nan,
            "n_paths": n, "session_lengths": sorted(session_lengths)}
