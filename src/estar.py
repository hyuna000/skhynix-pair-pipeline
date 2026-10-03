"""9절 비선형(ESTAR) 모형: KSS 버전의 임계값용.

모형 (윈도우의 z = (s - mu)/sigma, 연속 구간 안의 분만 사용):
    Δz_t = phi * z_{t-1} * (1 - exp(-gamma * z_{t-1}^2)) + sum_{j=1..p} c_j Δz_{t-j} + e_t
  - 0 근처에서는 거의 랜덤워크, 멀어질수록 복귀가 빨라진다 (KSS 검정의 대립가설과 같은 모양).
  - 안정 조건: 데이터가 사는 범위(|z| <= z_check, 기본 10)에서 1분 복귀율 -phi(1-exp(-gamma z^2)) 가 (0, 2).
    gamma 가 작으면 1 - exp(-g z^2) ~ g z^2 이라 phi*gamma (= KSS 의 delta, 3차항 계수) 만 식별되고
    phi 자체는 -2 보다 작게 나올 수 있다 (실제 데이터에서 흔함). 이때도 |z|<=10 에서 안정이면 그대로 쓴다.
    그 바깥은 시뮬레이션 폭주 점검(simulate 의 explode)이 맡는다.
  - Δz 래그는 호가 바운스(관측 잡음)가 만드는 음의 자기상관을 흡수한다. p 는 BIC 로 0..max_lags 에서 고른다.
추정: gamma 를 로그 격자에서 고정하면 나머지가 선형이므로 OLS (profile NLS), 격자 최적점 주변을 다시 세밀하게.
시뮬레이션: 적합 모형 재귀 + 적합 잔차 복원추출. 시작값은 윈도우의 실제 (z, Δz 래그) 에서 뽑는다.
"""
from __future__ import annotations

import numpy as np

from .gate_runner import segments_of


def _design(z_segs: list[np.ndarray], p: int):
    """연속 구간마다 (y=Δz_t, x=z_{t-1}, 래그 Δz_{t-1..t-p}) 를 쌓는다."""
    ys, xs, lags = [], [], []
    for z in z_segs:
        dz = np.diff(z)
        if len(dz) <= p + 1:
            continue
        t = np.arange(p, len(dz))
        ys.append(dz[t])
        xs.append(z[t])                       # z_{t-1} 에 해당 (dz[t] = z[t+1] - z[t])
        lags.append(np.column_stack([dz[t - j] for j in range(1, p + 1)]) if p else np.empty((len(t), 0)))
    if not ys:
        return None
    return np.concatenate(ys), np.concatenate(xs), np.vstack(lags)


def _ols_given_gamma(y, x, Lg, g):
    h = x * (1.0 - np.exp(-g * x * x))
    X = np.column_stack([h, Lg])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    return coef, r


def fit(z_segs: list[np.ndarray], max_lags: int = 10, gamma_bounds=(1e-5, 1e2), n_grid: int = 57,
        z_check: float = 10.0) -> dict:
    best = None
    grid = np.geomspace(*gamma_bounds, n_grid)
    for p in range(0, max_lags + 1):
        d = _design(z_segs, p)
        if d is None:
            break
        y, x, Lg = d
        n = len(y)
        ssr = []
        for g in grid:
            _, r = _ols_given_gamma(y, x, Lg, g)
            ssr.append(r @ r)
        i = int(np.argmin(ssr))
        # 세밀 탐색: 이웃 격자 사이 로그 공간
        lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, n_grid - 1)]
        fine = np.geomspace(lo, hi, 21)
        sf = [(lambda r: r @ r)(_ols_given_gamma(y, x, Lg, g)[1]) for g in fine]
        g = float(fine[int(np.argmin(sf))])
        coef, r = _ols_given_gamma(y, x, Lg, g)
        k = 2 + p                                  # phi, gamma, c_1..c_p
        bic = n * np.log(r @ r / n) + k * np.log(n)
        if best is None or bic < best["bic"]:
            best = {"bic": bic, "p": p, "gamma": g, "phi": float(coef[0]), "c": coef[1:].astype(float),
                    "resid": r, "n": n, "gamma_at_bound": bool(g <= gamma_bounds[0] * 1.0001 or g >= gamma_bounds[1] * 0.9999)}
    if best is None:
        return {"ok": False, "reason": "ESTAR_TOO_SHORT"}
    best["delta"] = best["phi"] * best["gamma"]
    rate_check = -best["phi"] * (1 - np.exp(-best["gamma"] * z_check ** 2))
    if not (best["phi"] < 0.0 and rate_check < 2.0):
        return {"ok": False, "reason": "ESTAR_NOT_MEAN_REVERTING" if best["phi"] >= 0 else "ESTAR_UNSTABLE_RANGE",
                "phi": best["phi"], "gamma": best["gamma"], "p": best["p"]}
    best["ok"] = True
    best["reason"] = ""
    best["z_segs"] = z_segs
    best["speed_at_2"] = float(-best["phi"] * (1 - np.exp(-4 * best["gamma"])))   # |z|=2 에서 1분 복귀율
    return best


def window_z(lnA: np.ndarray, lnL: np.ndarray, idx, a: float, b: float, mu: float, sigma: float,
             struct: bool, max_gap_min: int = 5) -> list[np.ndarray]:
    """윈도우 가격 -> 연속 구간별 z 배열. 공백(> max_gap_min 분)을 넘는 차분은 쓰지 않는다."""
    bb = 1.0 if struct else b
    s = lnA - bb * lnL - a
    z = (s - mu) / sigma
    return [z[i:j + 1] for i, j in segments_of(idx, max_gap_min) if j - i >= 2]


def simulate(model: dict, rng, n: int, L: int, explode: float = 25.0) -> tuple[np.ndarray, float]:
    """적합 ESTAR 경로 (n, L). 반환: 경로, 폭주 비율(|z| > explode 에 닿은 경로 비율)."""
    p, phi, g, c, res = model["p"], model["phi"], model["gamma"], model["c"], model["resid"]
    # 시작값: 실제 윈도우에서 (z_t, Δz_{t-1..t-p})
    starts = []
    for z in model["z_segs"]:
        dz = np.diff(z)
        for t in range(p, len(dz) + 1):
            starts.append((z[t], dz[t - p:t][::-1] if p else np.empty(0)))
    pick = rng.integers(0, len(starts), n)
    zcur = np.array([starts[i][0] for i in pick])
    lagm = np.array([starts[i][1] for i in pick]).reshape(n, p) if p else np.empty((n, 0))
    out = np.empty((n, L))
    out[:, 0] = zcur
    shocks = res[rng.integers(0, len(res), (n, L))]
    bad = np.zeros(n, bool)
    for t in range(1, L):
        dz = phi * zcur * (1 - np.exp(-g * zcur * zcur)) + shocks[:, t]
        if p:
            dz += lagm @ c
            lagm = np.column_stack([dz, lagm[:, :-1]]) if p > 1 else dz[:, None]
        zcur = zcur + dz
        big = np.abs(zcur) > explode
        if big.any():
            bad |= big
            zcur = np.clip(zcur, -explode, explode)
        out[:, t] = zcur
    return out, float(bad.mean())
