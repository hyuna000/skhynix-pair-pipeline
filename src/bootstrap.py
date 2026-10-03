"""부록 A 부트스트랩.

A-1 sieve wild (귀무 = 단위근)   : sieve_wild_unitroot(series, stat_fn, ...)
A-2 sieve wild KPSS (귀무 = 정상) : sieve_wild_kpss(series, ...)
A-4 VAR sieve wild (공적분 없는 I(1) 쌍) : var_sieve_pairs(y, x, ...) -> 재표본 (y*, x*) 생성기

모든 재표본은 B 개를 한꺼번에 만든다 (행 = 재표본). 가중치는 Rademacher.
p 값: 단위근 검정은 하측  p = (1 + #{tau* <= tau}) / (B + 1)
      KPSS 는 상측         p = (1 + #{eta* >= eta}) / (B + 1)
"""
from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def rademacher(rng, shape):
    return rng.integers(0, 2, size=shape) * 2.0 - 1.0


def ar_fit_aic(z: np.ndarray, pmax: int):
    """평균 제거된 z 에 AR(p), p = 0..pmax 를 같은 표본에서 AIC 로 선택. 반환 (phi, resid)."""
    z = np.asarray(z, float)
    T = len(z)
    pmax = int(min(pmax, T // 5))
    dep = z[pmax:]
    n = len(dep)
    if pmax > 0:
        X = np.column_stack([z[pmax - j: T - j] for j in range(1, pmax + 1)])
        Q, R = np.linalg.qr(X)
        qy = Q.T @ dep
    best = (np.log(dep @ dep / n), 0)
    rss = dep @ dep
    for p in range(1, pmax + 1):
        rss -= qy[p - 1] ** 2
        aic = np.log(max(rss, 1e-300) / n) + 2.0 * p / n
        if aic < best[0]:
            best = (aic, p)
    p = best[1]
    if p == 0:
        return np.zeros(0), z - z.mean()
    Xp = np.column_stack([z[p - j: T - j] for j in range(1, p + 1)])
    phi, *_ = np.linalg.lstsq(Xp, z[p:], rcond=None)
    e = z[p:] - Xp @ phi
    return phi, e


def _wild_innov(e, T, rng, B):
    """길이 T 의 wild 혁신 B 세트. 잔차가 T 보다 짧으면 앞부분을 잔차 앞쪽으로 채운다."""
    e = e - e.mean()
    if len(e) < T:
        e = np.r_[e[: T - len(e)], e]
    e = e[-T:]
    return rademacher(rng, (B, T)) * e[None, :]


def ar_simulate(phi, innov):
    a = np.r_[1.0, -np.asarray(phi)] if len(phi) else np.array([1.0])
    return lfilter([1.0], a, innov, axis=1)


def sieve_wild_unitroot(series, stat_fn, B, rng, pmax=30, return_draws=False):
    """A-1. stat_fn(series) -> 통계량(float). 재표본마다 래그 선택 포함 처음부터 계산."""
    s = np.asarray(series, float)
    tau = stat_fn(s)
    dx = np.diff(s)
    dx = dx - dx.mean()
    phi, e = ar_fit_aic(dx, pmax)
    innov = _wild_innov(e, len(dx), rng, B)
    dxs = ar_simulate(phi, innov)
    draws = np.empty(B)
    for b in range(B):
        sb = s[0] + np.r_[0.0, np.cumsum(dxs[b])]
        draws[b] = stat_fn(sb)
    p = (1 + np.sum(draws <= tau)) / (B + 1)
    out = {"stat": float(tau), "p_boot": float(p), "sieve_p": int(len(phi))}
    if return_draws:
        out["draws"] = draws
    return out


def sieve_wild_paths(series, B, rng, pmax=30):
    """A-1 귀무(단위근) 재표본 경로 B 개를 그대로 돌려준다 (여러 통계량이 같은 재표본을 공유할 때)."""
    s = np.asarray(series, float)
    dx = np.diff(s)
    dx = dx - dx.mean()
    phi, e = ar_fit_aic(dx, pmax)
    innov = _wild_innov(e, len(dx), rng, B)
    dxs = ar_simulate(phi, innov)
    paths = s[0] + np.concatenate([np.zeros((B, 1)), np.cumsum(dxs, axis=1)], axis=1)
    return paths, len(phi)


def sieve_wild_kpss(series, stat_fn, B, rng, pmax=30, cap=0.995, burn=300, return_draws=False):
    """A-2. 평균 제거 후 AR(p) 적합, 계수합 상한 cap, wild 재표본으로 정상 시리즈 재생성."""
    x = np.asarray(series, float)
    eta = stat_fn(x)
    z = x - x.mean()
    phi, e = ar_fit_aic(z, pmax)
    capped = False
    if len(phi) and phi.sum() > cap:
        phi = phi * (cap / phi.sum())
        capped = True
        p = len(phi)
        Xp = np.column_stack([z[p - j: len(z) - j] for j in range(1, p + 1)])
        e = z[p:] - Xp @ phi
    T = len(z)
    innov = _wild_innov(e, T, rng, B)
    burn_innov = rademacher(rng, (B, burn)) * rng.choice(e - e.mean(), size=(B, burn))
    sim = ar_simulate(phi, np.concatenate([burn_innov, innov], axis=1))[:, burn:]
    draws = np.array([stat_fn(sim[b]) for b in range(B)])
    p = (1 + np.sum(draws >= eta)) / (B + 1)
    out = {"stat": float(eta), "p_boot": float(p), "sieve_p": int(len(phi)), "ar_cap_applied": capped}
    if return_draws:
        out["draws"] = draws
    return out


# ---------------------------------------------------------------- A-4 ------
def var_fit_aic(W: np.ndarray, pmax: int):
    """W (T x 2) 평균 제거. VAR(p) p=0..pmax 를 같은 표본에서 AIC 선택. 반환 (A list, resid)."""
    T, k = W.shape
    pmax = int(min(pmax, T // 10))
    best = None
    for p in range(0, pmax + 1):
        Y = W[pmax:]
        if p > 0:
            X = np.column_stack([W[pmax - j: T - j] for j in range(1, p + 1)])
            C, *_ = np.linalg.lstsq(X, Y, rcond=None)
            E = Y - X @ C
        else:
            E = Y
        n = len(Y)
        S = E.T @ E / n
        aic = np.log(np.linalg.det(S)) + 2.0 * p * k * k / n
        if best is None or aic < best[0]:
            best = (aic, p)
    p = best[1]
    Y = W[p:]
    if p == 0:
        return [], Y - Y.mean(0)
    X = np.column_stack([W[p - j: T - j] for j in range(1, p + 1)])
    C, *_ = np.linalg.lstsq(X, Y, rcond=None)
    E = Y - X @ C
    A = [C[(j * k):(j + 1) * k].T for j in range(p)]      # W_t = sum A_j W_{t-j} + e
    return A, E


def var_sieve_pairs(y, x, B, rng, pmax=30):
    """A-4 (1)(2): 공적분 없는 I(1) 쌍 (y*, x*) 를 B 개 생성. 같은 eta 를 두 성분에 곱한다."""
    W = np.column_stack([np.diff(y), np.diff(x)])
    W = W - W.mean(0)
    A, E = var_fit_aic(W, pmax)
    E = E - E.mean(0)
    T = len(W)
    if len(E) < T:
        E = np.vstack([E[: T - len(E)], E])
    E = E[-T:]
    eta = rademacher(rng, (B, T))
    eps = eta[:, :, None] * E[None, :, :]            # (B, T, 2)
    p = len(A)
    Ws = np.zeros((B, T, 2))
    At = [a.T for a in A]
    for t in range(T):
        acc = eps[:, t, :].copy()
        for j in range(1, min(p, t) + 1):
            acc += Ws[:, t - j, :] @ At[j - 1]
        Ws[:, t, :] = acc
    ys = y[0] + np.concatenate([np.zeros((B, 1)), np.cumsum(Ws[:, :, 0], axis=1)], axis=1)
    xs = x[0] + np.concatenate([np.zeros((B, 1)), np.cumsum(Ws[:, :, 1], axis=1)], axis=1)
    return ys, xs, {"var_p": p}
