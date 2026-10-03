"""7절 Step 1~4 와 게이트 판정.

step1(lnL, lnA, cfg, rng_factory)          -> 두 로그 가격 각각 I(1) 인지 (네 조건)
step3_all(lnA, lnL, cfg, rng_factory)      -> 후보 4종의 b 추정(Step 2) + 스프레드 ADF/KSS + 게이트
step4_params(spread)                        -> mu, sigma, phi, theta, half-life (분)

rng_factory(stream_name) 는 이름별 독립 난수를 돌려준다. 같은 거래일이면 V1·V2 가 같은
스트림을 받는다 (공통 난수, A-5 (8)).
"""
from __future__ import annotations

import numpy as np
from statsmodels.tsa.adfvalues import mackinnonp

from . import bootstrap as bs
from . import hedge
from . import unitroot as ur


# ------------------------------------------------------------------ Step 1 --
def _kpss_table_p(stat):
    """KPSS(상수) 점근 표 p 값, 0.01~0.10 사이 선형보간 (statsmodels 와 같은 방식, 범위 밖은 끝값)."""
    cv = np.array([0.347, 0.463, 0.574, 0.739])
    pv = np.array([0.10, 0.05, 0.025, 0.01])
    return float(np.interp(stat, cv, pv))


def _ers_table_p(stat):
    """ADF-GLS(상수) 점근 표 p 값, 0.01~0.10 사이 선형보간 (ERS 임계값 -2.58/-1.95/-1.62)."""
    return float(np.interp(stat, [-2.58, -1.95, -1.62], [0.01, 0.05, 0.10]))


def _kss_table_p(stat):
    """KSS(2003) Table 1 case 2(평균 제거) 점근 임계값 -3.48/-2.93/-2.66 사이 선형보간."""
    return float(np.interp(stat, [-3.48, -2.93, -2.66], [0.01, 0.05, 0.10]))


def step1_series(z, cfg, rng_factory, name):
    c1 = cfg["step1"]
    crit = cfg["lags"]["criterion"]
    B = cfg["bootstrap"]["B"]
    pmax = cfg["lags"]["sieve_pmax"]
    crit_d = c1.get("diff_lag_criterion", crit)       # 차분 시리즈(조건 c)의 래그 규칙
    adfgls = lambda s: ur.adfgls_stat(s, cbar=c1["gls_cbar"], criterion=crit)["stat"]
    adfgls_d = lambda s: ur.adfgls_stat(s, cbar=c1["gls_cbar"], criterion=crit_d)["stat"]
    kpss = lambda s: ur.kpss_stat(s)["stat"]
    dz = np.diff(z)

    a = bs.sieve_wild_unitroot(z, adfgls, B, rng_factory(f"s1_{name}_a"), pmax)
    b = bs.sieve_wild_kpss(z, kpss, B, rng_factory(f"s1_{name}_b"), pmax, cap=c1["kpss_ar_sum_cap"])
    c = bs.sieve_wild_unitroot(dz, adfgls_d, B, rng_factory(f"s1_{name}_c"), pmax)
    d = bs.sieve_wild_kpss(dz, kpss, B, rng_factory(f"s1_{name}_d"), pmax, cap=c1["kpss_ar_sum_cap"])
    a["k"] = ur.adfgls_stat(z, cbar=c1["gls_cbar"], criterion=crit)["k"]
    c["k"] = ur.adfgls_stat(dz, cbar=c1["gls_cbar"], criterion=crit_d)["k"]
    c["lag_criterion"] = crit_d
    for r, s in ((b, z), (d, dz)):
        r["k"] = ur.kpss_stat(s)["k"]

    # (b) 판정 방식: "bootstrap"(A-2, 계획서) | "table"(KPSS 점근 임계값).
    # table 은 지속성 높은 정상 시리즈에서 과대기각하지만, (b) 에서 과대기각 = 통과가 늘어나는 쪽이라
    # 참 I(1) 오탈락을 줄인다. 시뮬레이션: 참 I(1) 통과율 bootstrap(cap .995) 0.44 -> table 0.99.
    # (c) 판정 방식: "bootstrap"(A-1, 계획서) | "table"(ERS 점근 임계값).
    # 차분 시리즈 부트스트랩은 2차 차분에 AR sieve 를 맞춰 귀무 분포가 왜곡된다(README).
    # pvalue_mode: "plan" = 조건별 스위치(kpss_b_method, adfgls_c_method) 사용,
    #              "bootstrap" = 네 조건 모두 부트스트랩, "table" = 네 조건 모두 점근 표.
    mode = c1.get("pvalue_mode", "plan")
    a["p_table"], c["p_table"] = _ers_table_p(a["stat"]), _ers_table_p(c["stat"])
    b["p_table"], d["p_table"] = _kpss_table_p(b["stat"]), _kpss_table_p(d["stat"])
    methods = {"a": "bootstrap", "b": c1.get("kpss_b_method", "bootstrap"),
               "c": c1.get("adfgls_c_method", "bootstrap"), "d": c1.get("kpss_d_method", "bootstrap")}
    if mode in ("bootstrap", "table"):
        methods = {k: mode for k in methods}
    for k, r in zip("abcd", (a, b, c, d)):
        r["method"] = methods[k]
        r["p_used"] = r["p_table"] if methods[k] == "table" else r["p_boot"]
    ok_a = a["p_used"] >= c1["alpha_a_level_adfgls"]     # 비기각
    ok_b = b["p_used"] < c1["alpha_b_level_kpss"]        # 기각
    ok_c = c["p_used"] < c1["alpha_c_diff_adfgls"]       # 기각
    ok_d = d["p_used"] >= c1["alpha_d_diff_kpss"]        # 비기각
    use = c1.get("gate_conditions", "abcd")              # 게이트에 쓰는 조건 (나머지는 기록만)
    fails = [k for k, ok in zip("abcd", (ok_a, ok_b, ok_c, ok_d)) if not ok and k in use]
    return {
        "level_adfgls": a, "level_kpss": b, "diff_adfgls": c, "diff_kpss": d,
        "cond": {"a": ok_a, "b": ok_b, "c": ok_c, "d": ok_d},
        "is_I1": not fails, "fail_conditions": "".join(fails),
    }


def step1(lnL, lnA, cfg, rng_factory):
    res = {}
    for name, z in (("L", lnL), ("A", lnA)):
        if z is None or len(z) == 0 or np.all(np.isnan(z)):
            res[name] = None
            continue
        res[name] = step1_series(np.asarray(z, float), cfg, rng_factory, name)
    avail = [r for r in res.values() if r is not None]
    passed = len(avail) == 2 and all(r["is_I1"] for r in avail)
    reasons = []
    for name, r in res.items():
        if r is None:
            reasons.append(f"{name}:NO_DATA")
        elif not r["is_I1"]:
            reasons.append(f"{name}:fail_{r['fail_conditions']}")
    return {"pass": passed, "reason": ";".join(reasons) or "I1_BOTH", "series": res}


# ------------------------------------------------------------- Step 2·3 -----
def _spread_stats(s, crit):
    s = s - s.mean()
    adf = ur.adf_stat(s, criterion=crit)
    kss = ur.kss_stat(s, criterion=crit)
    return adf, kss


def step3_all(lnA, lnL, cfg, rng_factory):
    """후보 4종에 대해 Step 2(b) + Step 3(ADF, KSS) + 게이트. 반환 dict[candidate] -> 결과."""
    c3 = cfg["step3"]
    crit = cfg["lags"]["criterion"]
    B = cfg["bootstrap"]["B"]
    pmax = cfg["lags"]["sieve_pmax"]
    Kmax = c3["dols_max_leads_lags"]
    alpha = c3["alpha"]
    y, x = np.asarray(lnA, float), np.asarray(lnL, float)

    out = {}
    for cand in hedge.CANDIDATES:
        a, b, extra = hedge.estimate(y, x, cand, Kmax)
        s = hedge.spread(y, x, a, b)
        adf, kss = _spread_stats(s, crit)
        lo, hi = hedge.hac_ci(y, x, cand, a, b, extra) if cand in ("ols", "dols") else (np.nan, np.nan)
        out[cand] = {"a": a, "b": b, "b_ci_lo": lo, "b_ci_hi": hi, "extra": extra,
                     "adf": adf, "kss": kss, "spread": s}

    # --- 후보 1~3: A-4 VAR sieve, 같은 재표본을 세 추정량이 공유 (임계값은 따로)
    ys, xs, vinfo = bs.var_sieve_pairs(y, x, B, rng_factory("s3_var"), pmax)
    draws = {c: {"adf": np.empty(B), "kss": np.empty(B)} for c in ("ols", "tls", "dols")}
    for i in range(B):
        for cand in ("ols", "tls", "dols"):
            a_, b_, _ = hedge.estimate(ys[i], xs[i], cand, Kmax)
            s_ = hedge.spread(ys[i], xs[i], a_, b_)
            if not np.isfinite(s_).all():
                draws[cand]["adf"][i] = draws[cand]["kss"][i] = np.nan
                continue
            ad, ks = _spread_stats(s_, crit)
            draws[cand]["adf"][i] = ad["stat"]
            draws[cand]["kss"][i] = ks["stat"]

    def pval(draw, stat):
        d = draw[np.isfinite(draw)]
        return float((1 + np.sum(d <= stat)) / (len(d) + 1))

    for cand in ("ols", "tls", "dols"):
        r = out[cand]
        r["adf_p_boot"] = pval(draws[cand]["adf"], r["adf"]["stat"])
        r["kss_p_boot"] = pval(draws[cand]["kss"], r["kss"]["stat"])
        r["adf_cv_boot"] = float(np.nanquantile(draws[cand]["adf"], alpha))
        r["kss_cv_boot"] = float(np.nanquantile(draws[cand]["kss"], alpha))
        r["var_p"] = vinfo["var_p"]
        r["adf_p_table"] = float(mackinnonp(r["adf"]["stat"], regression="c", N=2)) if cand == "ols" else np.nan
        r["adf_p_source"] = "mackinnon_eg" if (cand == "ols" and c3["ols_pvalue_source"] == "mackinnon") else "bootstrap_A4"
        r["adf_p"] = r["adf_p_table"] if r["adf_p_source"] == "mackinnon_eg" else r["adf_p_boot"]
        r["kss_p"] = r["kss_p_boot"]

    # --- 후보 4 (b=1): 단변량 A-1. ADF·KSS 가 같은 재표본을 공유 (min-p 결합 검정에 필요)
    r = out["struct"]
    s = r["spread"] - r["spread"].mean()
    paths, sieve_p = bs.sieve_wild_paths(s, B, rng_factory("s3_struct"), pmax)
    draws["struct"] = {"adf": np.empty(B), "kss": np.empty(B)}
    for i in range(B):
        ad, ks = _spread_stats(paths[i], crit)
        draws["struct"]["adf"][i], draws["struct"]["kss"][i] = ad["stat"], ks["stat"]
    r["adf_p_boot"] = pval(draws["struct"]["adf"], r["adf"]["stat"])
    r["kss_p_boot"] = pval(draws["struct"]["kss"], r["kss"]["stat"])
    r["adf_cv_boot"] = float(np.quantile(draws["struct"]["adf"], alpha))
    r["kss_cv_boot"] = float(np.quantile(draws["struct"]["kss"], alpha))
    r["var_p"] = sieve_p
    r["adf_p_table"] = float(mackinnonp(r["adf"]["stat"], regression="c", N=1))
    r["adf_p_source"] = "bootstrap_A1" if c3["struct_pvalue_source"] == "bootstrap" else "mackinnon_df"
    r["adf_p"] = r["adf_p_boot"] if r["adf_p_source"] == "bootstrap_A1" else r["adf_p_table"]
    r["kss_p"] = r["kss_p_boot"]

    # --- min-p 결합 검정: 같은 재표본에서 (p_ADF*, p_KSS*) 의 최솟값 분포로 p 값
    for cand, r in out.items():
        da, dk = draws[cand]["adf"], draws[cand]["kss"]
        ok = np.isfinite(da) & np.isfinite(dk)
        da, dk = da[ok], dk[ok]
        n = len(da)
        rank_a = (1 + (da[None, :] <= da[:, None]).sum(1)) / (n + 1)
        rank_k = (1 + (dk[None, :] <= dk[:, None]).sum(1)) / (n + 1)
        minp_null = np.minimum(rank_a, rank_k)
        minp_obs = min(r["adf_p_boot"], r["kss_p_boot"])
        r["minp_p"] = float((1 + np.sum(minp_null <= minp_obs)) / (n + 1))

    # --- p 값 출처 모드: plan(계획서 혼합) | bootstrap(전부 부트스트랩) | table(전부 점근 표)
    #     table: ADF 는 MacKinnon (후보 1~3 은 EG N=2 — TLS·DOLS 엔 근사), KSS 는 KSS(2003) case 2 표
    mode = c3.get("pvalue_mode", "plan")
    for cand, r in out.items():
        if cand != "ols":
            r["adf_p_table"] = float(mackinnonp(r["adf"]["stat"], regression="c", N=1 if cand == "struct" else 2))
        r["kss_p_table"] = _kss_table_p(r["kss"]["stat"])
        r["kss_p_source"] = "bootstrap"
        if mode == "bootstrap":
            r["adf_p"], r["adf_p_source"] = r["adf_p_boot"], "bootstrap"
        elif mode == "table":
            r["adf_p"], r["adf_p_source"] = r["adf_p_table"], ("mackinnon_df" if cand == "struct" else "mackinnon_eg")
            r["kss_p"], r["kss_p_source"] = r["kss_p_table"], "kss_table"

    # --- 게이트 판정. 규칙별로 모두 계산해 두고, 로그에는 설정한 규칙마다 한 줄씩 남긴다.
    #   or   : 7절 원안 표 (ADF 또는 KSS 5%)
    #   adf  : ADF 5% 단독
    #   kss  : KSS 5% 단독
    #   minp : min(p_ADF, p_KSS) 결합 부트스트랩 5% (두 검정 모두 부트스트랩 p 사용)
    for cand, r in out.items():
        adf_rej = r["adf_p"] < alpha
        kss_rej = r["kss_p"] < alpha
        r["adf_reject"], r["kss_reject"] = bool(adf_rej), bool(kss_rej)
        if adf_rej and kss_rej:
            cls_or = "BOTH_DEFAULT_LINEAR"
        elif adf_rej:
            cls_or = "LINEAR"
        elif kss_rej:
            cls_or = "NONLINEAR"
        else:
            cls_or = "NONE"
        minp_pass = r["minp_p"] < alpha
        r["gates"] = {
            "or": (cls_or != "NONE", cls_or),
            "adf": (bool(adf_rej), "LINEAR" if adf_rej else "NONE"),
            "kss": (bool(kss_rej), "NONLINEAR" if kss_rej else "NONE"),
            "minp": (bool(minp_pass), ("LINEAR" if r["adf_p_boot"] <= r["kss_p_boot"] else "NONLINEAR")
                     if minp_pass else "NONE"),
        }
        r["gate"], r["gate_class"] = r["gates"]["or"]
    return out


# ------------------------------------------------------------------ Step 4 --
def _hl(phi):
    if 0 < phi < 1:
        th = -np.log(phi)
        return float(th), float(np.log(2) / th)
    return np.nan, np.inf


def step4_params(s, method: str = "arma11"):
    """평균·표준편차·복귀속도·반감기.

    ar1    : s_t = c + phi s_{t-1} + e  (호가 잡음이 있으면 phi 가 0 쪽으로 편향 -> 반감기 과소추정)
    arma11 : s_t 를 ARMA(1,1) 로 MLE.  AR(1) 스프레드 + 백색 관측잡음 = ARMA(1,1) 이므로
             AR 계수가 잡음에 편향되지 않는다. 실패하면 ar1 로 대체하고 표시.
    두 값 모두 기록하고, method 로 고른 값을 반감기 필터에 쓴다.
    """
    import warnings
    s = np.asarray(s, float)
    mu, sigma = float(s.mean()), float(s.std(ddof=1))
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    (_, phi_ar1), *_ = np.linalg.lstsq(X, s[1:], rcond=None)
    th_ar1, hl_ar1 = _hl(phi_ar1)
    out = {"mu": mu, "sigma": sigma, "phi_ar1": float(phi_ar1), "half_life_ar1_min": hl_ar1,
           "phi_arma": np.nan, "ma_arma": np.nan, "half_life_arma_min": np.nan}
    if method == "arma11":
        try:
            from statsmodels.tsa.arima.model import ARIMA
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r = ARIMA(s - mu, order=(1, 0, 1), trend="n").fit()
            phi_a, ma_a = float(r.arparams[0]), float(r.maparams[0])
            out["phi_arma"], out["ma_arma"] = phi_a, ma_a
            out["half_life_arma_min"] = _hl(phi_a)[1]
            out["phi"], (out["theta"], out["half_life_min"]) = phi_a, _hl(phi_a)
            out["half_life_method"] = "arma11"
            return out
        except Exception:
            pass
    out["phi"], out["theta"], out["half_life_min"] = float(phi_ar1), th_ar1, hl_ar1
    out["half_life_method"] = "ar1" if method == "ar1" else "ar1_fallback"
    return out
