"""14절 크기·검정력 시뮬레이션 (warp-speed, Giacomini-Politis-White 2013).

반복마다 실제 통계량 1개 + 부트스트랩 재표본 1개만 계산하고, 재표본 통계량들의 분위수를
공통 임계값으로 써서 기각률을 낸다. 실제 데이터 검정에 쓰는 함수(src.unitroot, src.bootstrap)를 그대로 쓴다.

  python simulate.py                 # 기본: T = 1440, 2880 / R = 500
  python simulate.py --R 200 --T 1440 2880 4320

출력: reports/sim_step1.csv, reports/sim_step3.csv
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import bootstrap as bs
from src import unitroot as ur
from src.config import ROOT

# ------------------------------------------------------------------ DGP ----
def garch(rng, n):
    h, r, z = 1.0, np.empty(n), rng.standard_normal(n)
    for t in range(n):
        if t:
            h = 0.05 + 0.10 * r[t - 1] ** 2 + 0.85 * h
        r[t] = np.sqrt(h) * z[t]
    return r


def ar1(rng, n, hl, scale, burn=500):
    phi = np.exp(-np.log(2) / hl)
    e = scale * garch(rng, n + burn)
    x = np.zeros(n + burn)
    for t in range(1, n + burn):
        x[t] = phi * x[t - 1] + e[t]
    return x[burn:]


def price_dgp(kind, hl, T, rng):
    """Step 1 용 로그가격: I(1) 또는 박스권(정상 AR(1)), 호가 바운스 포함."""
    if kind == "I1":
        m = np.cumsum(0.0006 * garch(rng, T))
    else:
        m = ar1(rng, T, hl, 0.0006)
    return m + 0.0002 * rng.standard_normal(T)


def spread_dgp(kind, hl, T, rng):
    """Step 3 용 스프레드: 랜덤워크(귀무) 또는 OU(반감기 hl), 바운스 잡음 포함."""
    if kind == "RW":
        s = np.cumsum(0.0003 * garch(rng, T))
    else:
        phi = np.exp(-np.log(2) / hl)
        s = ar1(rng, T, hl, 0.003 * np.sqrt(1 - phi * phi))
    return s + 0.0003 * rng.standard_normal(T)


# --------------------------------------------------------------- 셀 계산 ----
ERS_1 = -2.58
KPSS_10 = 0.347
MACK_DF_5 = -2.862     # MacKinnon(2010) 상수, N=1, 점근 5%
KSS_5 = -2.93          # KSS(2003) case 2 (평균 제거) 5%


def cell_step1(args):
    kind, hl, T, R, seed = args
    rng = np.random.default_rng(seed)
    adfgls = lambda s: ur.adfgls_stat(s, criterion="rmaic")["stat"]
    kpss = lambda s: ur.kpss_stat(s)["stat"]
    a_stat, a_draw, b_stat, b_draw = (np.empty(R) for _ in range(4))
    for r in range(R):
        z = price_dgp(kind, hl, T, rng)
        ra = bs.sieve_wild_unitroot(z, adfgls, 1, rng, 30, return_draws=True)
        rb = bs.sieve_wild_kpss(z, kpss, 1, rng, 30, cap=0.995, return_draws=True)
        a_stat[r], a_draw[r] = ra["stat"], ra["draws"][0]
        b_stat[r], b_draw[r] = rb["stat"], rb["draws"][0]
    # (a) 통과 = 1% 비기각,  (b) 통과 = 10% 기각
    a_pass_boot = np.mean(a_stat > np.quantile(a_draw, 0.01))
    a_pass_tab = np.mean(a_stat > ERS_1)
    b_pass_boot = np.mean(b_stat > np.quantile(b_draw, 0.90))
    b_pass_tab = np.mean(b_stat > KPSS_10)
    ab_boot = np.mean((a_stat > np.quantile(a_draw, 0.01)) & (b_stat > np.quantile(b_draw, 0.90)))
    ab_tab = np.mean((a_stat > ERS_1) & (b_stat > KPSS_10))
    return {"dgp": "I(1)" if kind == "I1" else f"박스권 hl={hl}", "T": T, "R": R,
            "a_pass_boot": a_pass_boot, "a_pass_table": a_pass_tab,
            "b_pass_boot": b_pass_boot, "b_pass_table": b_pass_tab,
            "ab_pass_boot": ab_boot, "ab_pass_table": ab_tab}


def cell_step3(args):
    kind, hl, T, R, seed = args
    rng = np.random.default_rng(seed)
    adf = lambda s: ur.adf_stat(s - s.mean(), criterion="rmaic")["stat"]
    kss = lambda s: ur.kss_stat(s, criterion="rmaic")["stat"]
    A, Ad, K, Kd = (np.empty(R) for _ in range(4))
    for r in range(R):
        s = spread_dgp(kind, hl, T, rng)
        s = s - s.mean()
        # 한 번의 재표본에 두 통계량 (A-1, 귀무 = 단위근)
        dx = np.diff(s); dx -= dx.mean()
        phi, e = bs.ar_fit_aic(dx, 30)
        innov = bs._wild_innov(e, len(dx), rng, 1)
        sb = s[0] + np.r_[0.0, np.cumsum(bs.ar_simulate(phi, innov)[0])]
        A[r], K[r] = adf(s), kss(s)
        Ad[r], Kd[r] = adf(sb), kss(sb)
    rej_adf_b = A < np.quantile(Ad, 0.05)
    rej_kss_b = K < np.quantile(Kd, 0.05)
    rej_adf_t = A < MACK_DF_5
    rej_kss_t = K < KSS_5
    return {"dgp": "RW (귀무)" if kind == "RW" else f"OU hl={hl}", "T": T, "R": R,
            "adf_rej_boot": rej_adf_b.mean(), "adf_rej_table": rej_adf_t.mean(),
            "kss_rej_boot": rej_kss_b.mean(), "kss_rej_table": rej_kss_t.mean(),
            "gate_pass_boot": (rej_adf_b | rej_kss_b).mean(),
            "gate_pass_table": (rej_adf_t | rej_kss_t).mean()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=int, default=500)
    ap.add_argument("--T", type=int, nargs="+", default=[1440, 2880])
    ap.add_argument("--processes", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed", type=int, default=20261003)
    a = ap.parse_args(argv)
    s1 = [("I1", None)] + [("BOX", h) for h in (60, 240, 720)]
    s3 = [("RW", None)] + [("OU", h) for h in (30, 60, 120, 240)]
    jobs1 = [(k, h, T, a.R, a.seed + i) for i, ((k, h), T) in enumerate((x, T) for x in s1 for T in a.T)]
    jobs3 = [(k, h, T, a.R, a.seed + 100 + i) for i, ((k, h), T) in enumerate((x, T) for x in s3 for T in a.T)]
    with Pool(a.processes) as p:
        r1 = p.map(cell_step1, jobs1)
        r3 = p.map(cell_step3, jobs3)
    out = os.path.join(ROOT, "reports")
    os.makedirs(out, exist_ok=True)
    d1, d3 = pd.DataFrame(r1), pd.DataFrame(r3)
    d1.to_csv(os.path.join(out, "sim_step1.csv"), index=False, encoding="utf-8-sig")
    d3.to_csv(os.path.join(out, "sim_step3.csv"), index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 200)
    print("Step 1 통과율 ((a) 1% 비기각, (b) 10% 기각). I(1) 은 높을수록, 박스권은 낮을수록 좋음")
    print(d1.round(3).to_string(index=False))
    print("\nStep 3 기각률 (5%). RW 행 = 크기(0.05 근처가 정상), OU 행 = 검정력(높을수록 좋음)")
    print(d3.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
