"""7절 검정 통계량 (순수 함수: numpy 배열 -> 결과). 14절 시뮬레이션도 이 함수를 그대로 쓴다.

- adf_select : ADF 류 회귀 + 래그 선택(MAIC/RMAIC). 상수 포함 여부, 1차/3차(KSS) 회귀변수 선택.
- adf_stat   : 일반 ADF (상수), Step 3 선형.
- adfgls_stat: ERS ADF-GLS (상수만), Step 1.
- kss_stat   : Kapetanios-Shin-Snell (평균 제거, 상수 없음, y_{t-1}^3), Step 3 비선형.
- kpss_stat  : KPSS (상수), Bartlett 커널, Hobijn-Franses-Ooms 자동 대역폭, Step 1.

래그 선택
  MAIC  (Ng-Perron 2001):  ln s2_k + 2(tau_k + k)/(T - kmax),  tau_k = b0^2 sum y_{t-1}^2 / s2_k
  RMAIC = RSMAIC (Cavaliere, Phillips, Smeekes & Taylor 2015, Econometric Reviews 34(4), 식 (5)-(7)):
        1) OLS 로 추세(상수) 제거한 y^d 에 kmax 차 ADF 회귀 -> 잔차 e
        2) 변동성 경로 sigma_t = sqrt( NW 커널평활(e^2) ), 가우시안 커널, h = 0.1 (논문 시뮬레이션 값)
        3) 재척도 시리즈 y~_t = sum_{s<=t} dy^d_s / sigma_s
        4) y~ 에 MAIC 를 적용해 k 선택 -> 그 k 로 원래 시리즈의 검정 통계량 계산
        Perron-Qu(2007) 권고대로 선택은 OLS 추세제거 데이터에서 한다(GLS 검정이어도).
모든 k 는 같은 표본(t = kmax+1..T)에서 비교한다. 한 번의 QR 로 중첩 모형 전체를 계산.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import solve_triangular


def kmax_schwert(T: int) -> int:
    return int(np.floor(12 * (T / 100.0) ** 0.25))


def _design(y: np.ndarray, kmax: int):
    dy = np.diff(y)
    n = len(dy) - kmax
    dep = dy[kmax:]
    ylag = y[kmax:-1]
    if kmax > 0:
        lags = np.column_stack([dy[kmax - j: len(dy) - j] for j in range(1, kmax + 1)])
    else:
        lags = np.empty((n, 0))
    return dep, ylag, lags


def nw_volatility(e: np.ndarray, T: int, h: float = 0.1, offset: int = 0) -> np.ndarray:
    """잔차 e (시점 offset..offset+len(e)-1) 의 제곱을 가우시안 커널(대역폭 h, 시간 [0,1] 척도)로
    평활해 시점 0..T-1 전체의 변동성 sigma_t 를 돌려준다 (Nadaraya-Watson)."""
    from scipy.signal import fftconvolve
    x = np.zeros(T)
    w = np.zeros(T)
    x[offset:offset + len(e)] = e * e
    w[offset:offset + len(e)] = 1.0
    bw = max(1.0, h * T)
    L = int(min(T - 1, np.ceil(4 * bw)))
    k = np.exp(-0.5 * (np.arange(-L, L + 1) / bw) ** 2)
    num = fftconvolve(x, k, mode="same")
    den = fftconvolve(w, k, mode="same")
    v = num / np.maximum(den, 1e-300)
    v = np.maximum(v, 1e-300)
    return np.sqrt(v)


def rsmaic_select(y, kmax: int, h: float = 0.1) -> int:
    """Cavaliere et al.(2015) RSMAIC 로 래그 차수 선택."""
    y = np.asarray(y, float)
    yd = y - y.mean()                                   # OLS 추세제거 (상수)
    T = len(yd)
    dep, ylag, lags = _design(yd, kmax)
    X = np.column_stack([ylag] + ([lags] if lags.shape[1] else []))
    c, *_ = np.linalg.lstsq(X, dep, rcond=None)
    e = dep - X @ c                                     # 시점 kmax+1..T-1 (dy 인덱스 kmax..)
    dy = np.diff(yd)
    sig = nw_volatility(e, len(dy), h=h, offset=kmax)   # dy 각 시점의 변동성
    ytil = np.r_[0.0, np.cumsum(dy / sig)]
    ytil = ytil - ytil.mean()
    return adf_select(ytil, kmax=kmax, const=False, criterion="maic")["k"]


def adf_select(y, kmax=None, const=True, cube=False, criterion="rmaic", fixed_k=None):
    """ADF 류 회귀.  반환: dict(stat, k, b0, n)

    cube=True 이면 첫 회귀변수를 y_{t-1}^3 로 (KSS). 래그 선택은 항상 선형 회귀로 한 뒤
    (KSS 는 그 k 를 그대로 사용) 최종 통계량을 계산한다.
    """
    y = np.asarray(y, dtype=float)
    T = len(y)
    if kmax is None:
        kmax = kmax_schwert(T)
    kmax = int(min(kmax, max(0, T // 4)))
    dep, ylag, lags = _design(y, kmax)
    n = len(dep)

    def fit(first, kset):
        cols = ([np.ones(n)] if const else []) + [first] + ([lags] if lags.shape[1] else [])
        X = np.column_stack(cols)
        Q, R = np.linalg.qr(X)
        qy = Q.T @ dep
        p0 = 1 if const else 0
        m0 = p0 + 1
        e = dep - Q[:, :m0] @ qy[:m0]
        out = {}
        ys = first - first.mean() if const else first
        sy2 = ys @ ys
        # 상삼각 R 의 선행 블록의 역행렬 = R^-1 의 선행 블록 -> 한 번만 역행렬
        Rinv = solve_triangular(R, np.eye(R.shape[0]))
        for k in range(0, kmax + 1):
            m = m0 + k
            if k > 0:
                e = e - Q[:, m - 1] * qy[m - 1]
            if k not in kset:
                continue
            rss = e @ e
            row = Rinv[p0, :m]
            b0 = row @ qy[:m]
            # var(b0) = s2 * (R^-1 R^-T)[p0,p0] = s2 * ||row p0 of R^-1||^2
            v = row @ row
            s2df = rss / max(1, n - m)
            t = b0 / np.sqrt(s2df * v)
            s2 = rss / n
            if criterion == "bic":
                ic = np.log(s2) + k * np.log(n) / n
            elif criterion == "aic":
                ic = np.log(s2) + 2.0 * k / n
            else:   # maic
                tau = b0 * b0 * sy2 / s2
                ic = np.log(s2) + 2.0 * (tau + k) / n
            out[k] = (ic, t, b0)
        return out

    if fixed_k is not None:
        k_sel = int(min(fixed_k, kmax))
    elif criterion == "rmaic":
        k_sel = rsmaic_select(y, kmax)
    else:
        res = fit(ylag, set(range(kmax + 1)))
        k_sel = min(res, key=lambda k: res[k][0])
    first = ylag ** 3 if cube else ylag
    res2 = fit(first, {k_sel})
    _, t, b0 = res2[k_sel]
    return {"stat": float(t), "k": int(k_sel), "b0": float(b0), "n": int(n), "kmax": int(kmax)}


def adf_stat(y, criterion="rmaic", kmax=None):
    return adf_select(y, kmax=kmax, const=True, criterion=criterion)


def gls_demean(y, cbar=-7.0):
    y = np.asarray(y, dtype=float)
    T = len(y)
    a = 1.0 + cbar / T
    yt = np.r_[y[0], y[1:] - a * y[:-1]]
    zt = np.r_[1.0, np.full(T - 1, 1.0 - a)]
    beta = (zt @ yt) / (zt @ zt)
    return y - beta


def adfgls_stat(y, cbar=-7.0, criterion="rmaic", kmax=None):
    yd = gls_demean(y, cbar)
    return adf_select(yd, kmax=kmax, const=False, criterion=criterion)


def kss_stat(y, criterion="rmaic", kmax=None):
    yd = np.asarray(y, dtype=float)
    yd = yd - yd.mean()
    sel = adf_select(yd, kmax=kmax, const=False, criterion=criterion)
    out = adf_select(yd, kmax=kmax, const=False, cube=True, fixed_k=sel["k"])
    return out


def kpss_bandwidth(e: np.ndarray) -> int:
    """Hobijn, Franses & Ooms (1998) 자동 대역폭 (statsmodels 'auto' 와 같은 식)."""
    n = len(e)
    covlags = int(np.power(n, 2.0 / 9.0))
    s0 = e @ e / n
    s1 = 0.0
    for i in range(1, covlags + 1):
        rp = (e[i:] @ e[: n - i]) / (n / 2.0)
        s0 += rp
        s1 += i * rp
    if s0 <= 0:
        return 0
    s_hat = s1 / s0
    gamma_hat = 1.1447 * np.power(s_hat * s_hat, 1.0 / 3.0)
    return int(min(n - 1, gamma_hat * np.power(n, 1.0 / 3.0)))


def kpss_stat(x, nlags=None):
    x = np.asarray(x, dtype=float)
    e = x - x.mean()
    T = len(e)
    S = np.cumsum(e)
    l = kpss_bandwidth(e) if nlags is None else int(nlags)
    s = e @ e / T
    for j in range(1, l + 1):
        s += 2.0 * (1.0 - j / (l + 1.0)) * (e[j:] @ e[:-j]) / T
    return {"stat": float((S @ S) / (T * T * s)), "k": int(l), "n": int(T)}


# 점근 임계값 (로그 참고용)
KPSS_CV_C = {0.10: 0.347, 0.05: 0.463, 0.025: 0.574, 0.01: 0.739}
ERS_CV_C = {0.10: -1.62, 0.05: -1.95, 0.01: -2.58}       # ADF-GLS, 상수만 (DF 무상수 분포)
KSS_CV_DEMEANED = {0.10: -2.66, 0.05: -2.93, 0.01: -3.48}  # KSS 2003 Table 1, case 2 (참고만, 게이트엔 미사용)
