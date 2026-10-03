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

구간 배열 (선택, 길이 L): act[t]=False 면 그 분에는 주문 없음(정산 버퍼), entry_ok[t]=False 면 진입 없음(세션 마지막 분).
  fund_long[t]/fund_short[t] = 그 분 정산에서 롱/숏 스프레드가 내는 펀딩 (z 단위, 받으면 음수). 정산을 들고 넘길 때 쓴다.
  구간 마지막 분은 항상 강제 청산.

경로 생성 (model):
  "linear"          위 AR(1)+잡음 모형
  "block_bootstrap" 윈도우 실제 z 의 정상 블록 부트스트랩 (평균 블록 길이 block_len, 연속 구간 경계에서 새 블록)
                    -> 두꺼운 꼬리, 호가 바운스, 실제 자기상관을 모형 가정 없이 반영 (ADF 버전 기본)
  "estar"           src/estar.py 적합 모형 경로 (KSS 버전)
수준 이동: level_sd > 0 이면 경로마다 상수 N(0, level_sd^2) 를 더한다. 윈도우 모수로 본 다음 날 z 의 평균이
  0 에서 벗어나는 표본 밖 위험을 반영 (실측: 그날 z 평균이 -1.8 ~ +3.1). 이것이 없으면 기대 수익이 크게 부풀려진다.

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


def rules_vectorized(Z, entry, exit_, stop, c_entry, c_exit, c_forced, rearm=True,
                     act=None, entry_ok=None, fund_long=None, fund_short=None, fund_out=None):
    """Z (N paths, L minutes). 반환: 경로별 순손익(z 단위, 펀딩 포함), 거래 수.
    fund_out 을 주면 경로별 펀딩 지급액(z 단위)을 더해 준다."""
    N, L = Z.shape
    pos = np.zeros(N, np.int8)
    ez = np.zeros(N)
    pnl = np.zeros(N)
    ntr = np.zeros(N, np.int32)
    armed = np.ones(N, bool)
    for t in range(L):
        zt = Z[:, t]
        last = t == L - 1
        if fund_long is not None and (fund_long[t] != 0 or fund_short[t] != 0) and pos.any():
            paid = np.where(pos == 1, fund_long[t], np.where(pos == -1, fund_short[t], 0.0))
            pnl -= paid
            if fund_out is not None:
                fund_out += paid
        if act is not None and not act[t] and not last:
            continue
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
        if not last and (entry_ok is None or entry_ok[t]):
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


def block_bootstrap_paths(rng, z_segs: list, n: int, L: int, mean_block: float) -> np.ndarray:
    """정상 블록 부트스트랩 (Politis & Romano 1994). 매 분 확률 1/mean_block 로, 또는 구간 끝에서 새 시작점."""
    z = np.concatenate(z_segs)
    seg_end = np.concatenate([np.full(len(sg), i0 + len(sg) - 1) for sg, i0 in
                              zip(z_segs, np.cumsum([0] + [len(sg) for sg in z_segs[:-1]]))])
    N = len(z)
    pos = rng.integers(0, N, n)
    out = np.empty((n, L))
    p_new = 1.0 / max(mean_block, 1.0)
    for t in range(L):
        out[:, t] = z[pos]
        jump = (rng.random(n) < p_new) | (pos >= seg_end[pos])
        pos = np.where(jump, rng.integers(0, N, n), pos + 1)
    return out


def bertram_entry(theta: float, c_latent: float) -> float:
    """Bertram(2010): 진입 a(<0), 청산 m=-a, 단위시간 기대수익 최대. 잠재 z 단위 |a| 를 돌려준다."""
    if not (theta > 0) or not np.isfinite(c_latent):
        return np.nan
    a = np.linspace(0.05, 4.0, 400)
    et = (2 * np.pi / theta) * erfi(a / np.sqrt(2))
    mu = (2 * a - c_latent) / et
    return float(a[np.argmax(mu)])


def funding_z(cfg, b: float, candidate: str, sigma: float, rate_fc: dict, symbols: dict) -> dict:
    """정산 1회에 롱 스프레드(ADR 롱, 본주 숏)가 내는 펀딩, z 단위. 심볼별. 숏 스프레드는 부호 반대.
    달러 = C*wA*r_A (ADR 정산) 또는 -C*wL*r_L (본주 정산).  z 1 단위 = C/(1+b)*sigma 달러."""
    wA, wL = leg_weights(b, candidate)
    unit_over_C = sigma / (1.0 + (1.0 if candidate == "struct" else b))
    return {symbols["A"]: wA * rate_fc.get(symbols["A"], 0.0) / unit_over_C,
            symbols["L"]: -wL * rate_fc.get(symbols["L"], 0.0) / unit_over_C}


def optimize(cfg, params: dict, b: float, candidate: str, layout, symbols: dict, seed: int,
             model: str = "linear", estar_model: dict | None = None, rate_fc: dict | None = None,
             z_segs: list | None = None, level_sd: float = 0.0):
    """그 거래일의 보유 구간에 대해 하루 기대 순수익(펀딩 포함)이 최대인 (entry, exit, stop).

    layout: 구간 길이(int) 목록, 또는 dict 목록 {"L", "act", "entry_ok", "settle": [(t, symbol), ...]}
            (sessions.segment_layout). int 면 버퍼·정산 없이 구간 끝 강제 청산만 있는 단순 구간.
    model : "linear" (AR(1)+잡음) | "estar" (estar_model 경로)
    rate_fc: 심볼별 예상 펀딩률 (T0 이전 실현값으로 만든 것). None 이면 펀딩 0.
    """
    tc = cfg["thresholds"]
    sigma = params["sigma"]
    phi = params["phi"]
    phi1 = params.get("phi_ar1", phi)
    if not (0 < phi < 1):
        phi = phi1 if 0 < phi1 < 1 else np.nan
    if not (sigma > 0) or (model == "linear" and not np.isfinite(phi)):
        return {"ok": False, "reason": "BAD_PARAMS"}
    if model == "block_bootstrap" and (not z_segs or sum(len(x) for x in z_segs) < 60):
        return {"ok": False, "reason": "NO_WINDOW_Z"}
    sx2 = float(np.clip(phi1 / phi, 0.05, 1.0)) if (np.isfinite(phi) and 0 < phi1 < 1) else 1.0
    cz = cost_z(cfg, b, candidate, sigma, symbols)
    fz = funding_z(cfg, b, candidate, sigma, rate_fc or {}, symbols)
    rng = np.random.default_rng(seed)
    n = int(tc["n_paths"])
    segs = []
    for seg in layout:
        if isinstance(seg, (int, np.integer)):
            seg = {"L": int(seg), "act": None, "entry_ok": None, "settle": []}
        segs.append(seg)
    explode = 0.0
    for seg in segs:
        L = seg["L"]
        if model == "estar":
            from . import estar
            seg["Z"], fr = estar.simulate(estar_model, rng, n, L)
            explode = max(explode, fr)
        elif model == "block_bootstrap":
            seg["Z"] = block_bootstrap_paths(rng, z_segs, n, L, float(tc.get("block_len_min", 120)))
        else:
            seg["Z"] = simulate_paths(rng, n, L, phi, sx2)
        fl = np.zeros(L)
        for t, sym in seg["settle"]:
            if 0 <= t < L:
                fl[t] += fz.get(sym, 0.0)
        seg["fl"], seg["fs"] = (fl, -fl) if np.any(fl) else (None, None)
    if level_sd > 0:
        shift = rng.normal(0.0, level_sd, (n, 1))      # 같은 날의 모든 구간에 같은 이동
        for seg in segs:
            seg["Z"] = seg["Z"] + shift
    if model == "estar" and explode > float(tc.get("estar_max_explode_frac", 0.01)):
        return {"ok": False, "reason": "ESTAR_UNSTABLE", "explode_frac": explode}
    rearm = bool(cfg["trading"]["rearm_after_stop"])
    best = None
    for e, x, s in itertools.product(tc["entry_grid"], tc["exit_grid"], tc["stop_grid"]):
        if s <= e or x >= e:
            continue
        tot_z, tot_tr, tot_f = 0.0, 0.0, 0.0
        for seg in segs:
            fo = np.zeros(n)
            p, ntr = rules_vectorized(seg["Z"], e, x, s, cz["entry"], cz["exit"], cz["forced"], rearm,
                                      seg["act"], seg["entry_ok"], seg["fl"], seg["fs"], fo)
            tot_z += p.mean()
            tot_tr += ntr.mean()
            tot_f += fo.mean()
        if best is None or tot_z > best[0]:
            best = (tot_z, e, x, s, tot_tr, tot_f)
    tot_z, e, x, s, ntr, tf = best
    capital = cfg["trading"]["capital_usd"]
    unit = capital / (1.0 + (1.0 if candidate == "struct" else b)) * sigma   # z 1 단위의 달러 가치
    bz = np.nan
    if np.isfinite(phi):
        bz = bertram_entry(-np.log(phi), cz["entry"] * 2 / np.sqrt(sx2))
    out = {"ok": True, "model": model, "entry_z": e, "exit_z": x, "stop_z": s,
           "exp_pnl_usd": float(tot_z * unit), "exp_trades": float(ntr), "exp_funding_usd": float(tf * unit),
           "cost_roundtrip_z": float(cz["entry"] + cz["exit"]), "latent_var_share": sx2,
           "bertram_entry_z": float(bz * np.sqrt(sx2)) if np.isfinite(bz) else np.nan,
           "n_paths": n, "session_lengths": sorted(sg["L"] for sg in segs), "level_sd": float(level_sd)}
    if model == "estar":
        out["explode_frac"] = explode
    return out
