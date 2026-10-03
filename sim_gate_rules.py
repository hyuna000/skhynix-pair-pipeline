"""게이트 규칙 비교 시뮬레이션 (warp-speed). 구조적 후보(b=1) 스프레드, 기본 T=2880.

규칙
  OR    : ADF 5% 또는 KSS 5% (현재)
  BONF  : ADF 2.5% 또는 KSS 2.5% (본페로니)
  ADF   : ADF 5% 단독 (KSS 는 비선형 표시만)
  MINP  : min(p_ADF, p_KSS) 의 부트스트랩 귀무 분포로 5% 임계값을 정하는 결합 검정
DGP: 랜덤워크(크기), OU 반감기 60/120/240분,
     ESTAR (작은 괴리는 거의 안 돌아오고 큰 괴리만 강하게 복귀; KSS 를 넣은 이유)

  python sim_gate_rules.py --R 400
"""
import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from simulate import garch, spread_dgp
from src import bootstrap as bs
from src import unitroot as ur


def estar(rng, T, phi, band, burn=500):
    """s_t = s_{t-1} + phi * s_{t-1} * (1 - exp(-(s_{t-1}/band)^2)) + e_t"""
    e = 0.0002 * garch(rng, T + burn)
    s = np.zeros(T + burn)
    for t in range(1, T + burn):
        x = s[t - 1]
        s[t] = x + phi * x * (1 - np.exp(-(x / band) ** 2)) + e[t]
    return s[burn:] + 0.0001 * rng.standard_normal(T)


def gen(kind, rng, T):
    if kind == "RW":
        return spread_dgp("RW", None, T, rng)
    if kind.startswith("OU"):
        return spread_dgp("OU", int(kind[2:]), T, rng)
    if kind == "ESTAR_strong":
        return estar(rng, T, -0.05, 0.004)
    if kind == "ESTAR_weak":
        return estar(rng, T, -0.02, 0.006)
    raise ValueError(kind)


def cell(args):
    kind, T, R, seed = args
    rng = np.random.default_rng(seed)
    adf = lambda s: ur.adf_stat(s - s.mean(), criterion="rmaic")["stat"]
    kss = lambda s: ur.kss_stat(s, criterion="rmaic")["stat"]
    A, K, Ad, Kd = (np.empty(R) for _ in range(4))
    for r in range(R):
        s = gen(kind, rng, T)
        s = s - s.mean()
        dx = np.diff(s)
        dx -= dx.mean()
        phi, e = bs.ar_fit_aic(dx, 30)
        sb = s[0] + np.r_[0.0, np.cumsum(bs.ar_simulate(phi, bs._wild_innov(e, len(dx), rng, 1))[0])]
        A[r], K[r], Ad[r], Kd[r] = adf(s), kss(s), adf(sb), kss(sb)
    pa = lambda x: (1 + (Ad[None, :] <= np.atleast_1d(x)[:, None]).sum(1)) / (R + 1)
    pk = lambda x: (1 + (Kd[None, :] <= np.atleast_1d(x)[:, None]).sum(1)) / (R + 1)
    p_adf, p_kss = pa(A), pk(K)
    minp_null = np.minimum(pa(Ad), pk(Kd))
    c_minp = np.quantile(minp_null, 0.05)
    return {"DGP": kind,
            "OR (현재)": np.mean((p_adf < .05) | (p_kss < .05)),
            "BONF 각2.5%": np.mean((p_adf < .025) | (p_kss < .025)),
            "ADF 단독": np.mean(p_adf < .05),
            "MINP 결합5%": np.mean(np.minimum(p_adf, p_kss) <= c_minp),
            "KSS 단독": np.mean(p_kss < .05)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=400)
    ap.add_argument("--T", type=int, default=2880)
    a = ap.parse_args()
    kinds = ["RW", "OU60", "OU120", "OU240", "ESTAR_strong", "ESTAR_weak"]
    with Pool(os.cpu_count()) as p:
        res = p.map(cell, [(k, a.T, a.R, 900 + i) for i, k in enumerate(kinds)])
    df = pd.DataFrame(res)
    os.makedirs("reports", exist_ok=True)
    df.to_csv("reports/sim_gate_rules.csv", index=False, encoding="utf-8-sig")
    print(f"게이트 통과율 (T={a.T}, R={a.R}). RW 행 = 잘못 통과(5% 근처가 목표), 나머지 = 검정력")
    print(df.round(3).to_string(index=False))
