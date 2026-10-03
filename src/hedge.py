"""2절·7절 Step 2: 헤지비율 후보.  y = ln P_ADR, x = ln P_본주.

  ols    : y = a + b x + u
  tls    : 평균 제거 후 직교회귀 (공분산 행렬 최소 고유벡터)
  dols   : y = a + b x + sum_{j=-K..K} c_j dx_{t+j} + u,  K 는 BIC (0..Kmax). 윈도우 안의 리드만 사용.
  struct : b = 1, a = 윈도우 평균 (스프레드 = ln P_ADR + ln10 - ln P_본주 - 평균)

스프레드 s_t = y_t - b x_t - a (DOLS 도 Δ항 없이 이 식으로).
"""
from __future__ import annotations

import numpy as np

CANDIDATES = ("ols", "tls", "dols", "struct")


def est_ols(y, x):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1]), {}


def est_tls(y, x):
    mx, my = x.mean(), y.mean()
    C = np.cov(np.vstack([x - mx, y - my]))
    w, V = np.linalg.eigh(C)
    v = V[:, 0]                       # 최소 고유값의 고유벡터 (vx, vy)
    if abs(v[1]) < 1e-15:
        return float("nan"), float("nan"), {}
    b = -v[0] / v[1]
    return float(my - b * mx), float(b), {}


def _dols_design(y, x, K):
    dx = np.diff(x)                   # dx[i] = x[i+1]-x[i] = Δx_{i+1}
    T = len(y)
    idx = np.arange(K + 1, T - K)     # Δx_{t+j} = dx[t+j-1], j=-K..K 가 모두 존재하는 t
    cols = [np.ones(len(idx)), x[idx]]
    for j in range(-K, K + 1):
        cols.append(dx[idx + j - 1])
    return y[idx], np.column_stack(cols), idx


def est_dols(y, x, Kmax=5):
    # 같은 표본(Kmax 기준)에서 BIC 비교
    best = None
    T = len(y)
    lo, hi = Kmax + 1, T - Kmax
    for K in range(0, Kmax + 1):
        yy, X, idx = _dols_design(y, x, K)
        keep = (idx >= lo) & (idx < hi)
        yy, X = yy[keep], X[keep]
        coef, *_ = np.linalg.lstsq(X, yy, rcond=None)
        e = yy - X @ coef
        n, m = X.shape
        bic = n * np.log(e @ e / n) + m * np.log(n)
        if best is None or bic < best[0]:
            best = (bic, K, coef)
    _, K, coef = best
    return float(coef[0]), float(coef[1]), {"dols_K": int(K)}


def estimate(y, x, method, Kmax=5, ln_ratio=np.log(10.0)):
    """반환 (a, b, extra).  struct 는 a = mean(y + ln10 - x) - ln10 형태로 정리해 s = y - x - a."""
    if method == "ols":
        return est_ols(y, x)
    if method == "tls":
        return est_tls(y, x)
    if method == "dols":
        return est_dols(y, x, Kmax)
    if method == "struct":
        return float(np.mean(y - x)), 1.0, {}
    raise ValueError(method)


def spread(y, x, a, b):
    return y - b * x - a


def hac_ci(y, x, method, a, b, extra, level=0.95):
    """OLS·DOLS 의 b 에 대한 Newey-West HAC 신뢰구간 (기록용). TLS 는 미구현(NaN)."""
    import statsmodels.api as sm
    if method == "ols":
        X = sm.add_constant(x)
        yy = y
    elif method == "dols":
        yy, X, _ = _dols_design(y, x, extra.get("dols_K", 0))
    else:
        return float("nan"), float("nan")
    T = len(yy)
    L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
    r = sm.OLS(yy, X).fit(cov_type="HAC", cov_kwds={"maxlags": L})
    lo, hi = r.conf_int(1 - level)[1]
    return float(lo), float(hi)
