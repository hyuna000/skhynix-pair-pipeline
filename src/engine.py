"""9·10절 매매 엔진: 한 거래일 x 한 후보의 실제 가격 위 가상 매매.

규칙은 thresholds.rules_vectorized 와 같다 (tests/test_engine.py 에서 대조).
신호   : 분 m 의 신호가격(중간가, 없으면 체결가 종가)으로 z_m = (s_m - mu) / sigma
체결   : 오더북이 있으면 분 m 끝의 호가창 VWAP (fill_basis = book_vwap)
         없으면 분 m+1 시가 ± (반 스프레드 + extra) bp  (bar_open), 그것도 없으면 분 m 종가 (bar_close)
         강제 청산은 슬리피지 x forced_exit_slippage_mult
체결 시각 = 분 m + 60초.  펀딩: [진입 체결 - 15초, 청산 체결 + 15초] 안의 정산은 보유로 본다.
         펀딩률 기록이 없는 정산은 funding.missing_rate 로 가정하고 rate_source=MISSING_ASSUMED 로 남긴다.
세션 끝: end_reason HOLD 면 청산하지 않는다 (정산을 들고 넘김). 버퍼 분에는 주문을 내지 않는다.
손익   : 레그별 수량 x (청산가 - 진입가) x 방향.  수수료 = taker x 체결 명목가치 (4회).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .orderbook import book_vwap
from .thresholds import leg_weights

ONE = pd.Timedelta(minutes=1)


@dataclass
class MarketDay:
    """거래일 하나의 시장 데이터 (UTC 분 인덱스)."""
    px: pd.DataFrame                  # L, A (신호가격), L_is_mid, A_is_mid
    opens: pd.DataFrame               # L, A 분봉 시가
    closes: pd.DataFrame              # L, A 분봉 종가
    books: dict                       # key("L"/"A") -> DataFrame(index ts_utc, bid_px, bid_qty, ask_px, ask_qty)
    symbols: dict                     # {"L": sym, "A": sym}


@dataclass
class Leg:
    key: str
    side: int          # +1 매수(롱), -1 매도(숏)
    qty: float
    entry_px: float
    entry_ref: float
    entry_basis: str
    exit_px: float = np.nan
    exit_ref: float = np.nan
    exit_basis: str = ""


@dataclass
class Position:
    direction: int     # +1 롱 스프레드(ADR 롱, 본주 숏), -1 숏 스프레드
    entry_minute: pd.Timestamp
    entry_z: float
    session_idx: int
    legs: list = field(default_factory=list)
    mae: float = 0.0
    mfe: float = 0.0


def _fill(md: MarketDay, cfg: dict, key: str, minute: pd.Timestamp, side: int, notional: float,
          ref: float, forced: bool) -> tuple[float, str]:
    c = cfg["costs"]
    sym = md.symbols[key]
    book = md.books.get(key)
    if book is not None and minute in book.index:
        r = book.loc[minute]
        px, qty = (r["ask_px"], r["ask_qty"]) if side > 0 else (r["bid_px"], r["bid_qty"])
        n = int(c["book_depth_levels"])
        p, full = book_vwap(list(px)[:n], list(qty)[:n], notional)
        if np.isfinite(p):
            return float(p), "book_vwap" if full else "book_vwap_partial"
    slip = (c["half_spread_bp"].get(sym, 0.0) + c["extra_slippage_bp"]) * 1e-4
    if forced:
        slip *= c["forced_exit_slippage_mult"]
    nxt = minute + ONE
    if nxt in md.opens.index and np.isfinite(md.opens.at[nxt, key]):
        base, basis = md.opens.at[nxt, key], "bar_open"
    elif minute in md.closes.index and np.isfinite(md.closes.at[minute, key]):
        base, basis = md.closes.at[minute, key], "bar_close"
    else:
        base, basis = ref, "signal_price"
    return float(base * (1 + side * slip)), basis


def run_day(cfg, md: MarketDay, sessions, params: dict, thr: dict, candidate: str, sched, day_minutes,
            trading_date: str) -> dict:
    """반환: trades(list of dict), signals(list of dict), funding(list of dict), marks."""
    a, b, mu, sigma = params["a"], params["b"], params["mu"], params["sigma"]
    e, x, s_stop = thr["entry_z"], thr["exit_z"], thr["stop_z"]
    wA, wL = leg_weights(b, candidate)
    bb = 1.0 if candidate == "struct" else b
    C = cfg["trading"]["capital_usd"]
    step = cfg["trading"]["qty_step"]
    fee = cfg["costs"]["taker_fee"]
    rearm = bool(cfg["trading"]["rearm_after_stop"])
    hold_win = pd.Timedelta(seconds=cfg["funding"]["holding_window_sec"])
    miss_rate = float(cfg["funding"].get("missing_rate", 0.0))
    fee_alt = float(cfg["costs"].get("fee_alt", fee))

    sess_of = {}
    for ss in sessions:
        for m in pd.date_range(ss.start, ss.end, freq="1min"):
            sess_of[m.tz_convert("UTC")] = ss
    px = md.px
    trades, signals, fund_rows = [], [], []
    pos: Position | None = None
    armed = True
    episode, in_episode = 0, False
    pending_forced = None   # 데이터 공백으로 세션 끝에서 못 닫은 경우

    def zval(m):
        if m not in px.index:
            return np.nan, np.nan, np.nan
        L, A = px.at[m, "L"], px.at[m, "A"]
        if not (np.isfinite(L) and np.isfinite(A)):
            return np.nan, L, A
        sp = np.log(A) - bb * np.log(L) - a if candidate != "struct" else np.log(A) - np.log(L) - a
        return (sp - mu) / sigma, L, A

    def close_pos(m, z, L, A, reason, forced, flags=""):
        nonlocal pos
        refs = {"L": L, "A": A}
        for lg in pos.legs:
            px_, basis = _fill(md, cfg, lg.key, m, -lg.side, lg.qty * refs[lg.key], refs[lg.key], forced)
            lg.exit_px, lg.exit_ref, lg.exit_basis = px_, refs[lg.key], basis
        t_in = pos.entry_minute + ONE
        t_out = m + ONE
        gross = sum(lg.qty * (lg.exit_px - lg.entry_px) * lg.side for lg in pos.legs)
        gross_ref = sum(lg.qty * (lg.exit_ref - lg.entry_ref) * lg.side for lg in pos.legs)
        fees = sum(fee * lg.qty * (lg.entry_px + lg.exit_px) for lg in pos.legs)
        fees_alt = sum(fee_alt * lg.qty * (lg.entry_px + lg.exit_px) for lg in pos.legs)
        # 펀딩 (정상이라면 0 건)
        fsum, nf, nmiss = 0.0, 0, 0
        for st, sym in sched.settlements(t_in - hold_win, t_out + hold_win):
            if not (t_in - hold_win <= st <= t_out + hold_win):
                continue
            key = "L" if sym == md.symbols["L"] else "A"
            lg = [q for q in pos.legs if q.key == key][0]
            rate = sched.rate(sym, st)
            src = "data"
            if not np.isfinite(rate):
                rate, src = miss_rate, "MISSING_ASSUMED"
                nmiss += 1
            mark = px.at[st.floor("min"), key] if st.floor("min") in px.index else lg.entry_ref
            notional = lg.qty * mark
            amt = -lg.side * notional * rate   # 양수 펀딩률: 롱이 지급
            fund_rows.append({"trading_date": trading_date, "candidate": candidate,
                              "settle_utc": str(st), "symbol": sym, "leg_side": lg.side, "rate": rate,
                              "notional": notional, "mark_basis": "signal_price", "amount": amt, "rate_source": src,
                              "trade_entry_utc": str(t_in)})
            fsum += amt
            nf += 1
        legA = [q for q in pos.legs if q.key == "A"][0]
        legL = [q for q in pos.legs if q.key == "L"][0]
        trades.append({
            "trading_date": trading_date, "candidate": candidate, "session_idx": pos.session_idx,
            "direction": "LONG_SPREAD" if pos.direction > 0 else "SHORT_SPREAD",
            "entry_signal_utc": str(pos.entry_minute), "entry_fill_utc": str(t_in),
            "exit_signal_utc": str(m), "exit_fill_utc": str(t_out),
            "holding_min": int((t_out - t_in) / ONE),
            "entry_z": pos.entry_z, "exit_z": z, "exit_type": reason,
            "A_qty": legA.qty, "A_side": legA.side, "A_entry_px": legA.entry_px, "A_exit_px": legA.exit_px,
            "A_entry_basis": legA.entry_basis, "A_exit_basis": legA.exit_basis,
            "L_qty": legL.qty, "L_side": legL.side, "L_entry_px": legL.entry_px, "L_exit_px": legL.exit_px,
            "L_entry_basis": legL.entry_basis, "L_exit_basis": legL.exit_basis,
            "entry_notional": sum(q.qty * q.entry_px for q in pos.legs),
            "gross_pnl_ref": gross_ref, "gross_pnl": gross, "slippage": gross_ref - gross, "fees": fees,
            "funding": -fsum, "funding_events": nf, "funding_missing": nmiss,
            "net_pnl": gross - fees + fsum,
            "fees_alt": fees_alt, "net_pnl_fee_alt": gross - fees_alt + fsum,
            "net_pnl_cost1_5": gross_ref - 1.5 * ((gross_ref - gross) + fees) + fsum,
            "net_pnl_cost2": gross_ref - 2.0 * ((gross_ref - gross) + fees) + fsum,
            "mae": pos.mae, "mfe": pos.mfe, "flags": flags,
        })
        pos = None

    for m in day_minutes:                     # 거래일 전체 분 (UTC), 세션 밖 포함
        ss = sess_of.get(m)
        z, L, A = zval(m)
        if pos is not None and np.isfinite(z):
            mtm = sum(lg.qty * ((L if lg.key == "L" else A) - lg.entry_px) * lg.side for lg in pos.legs)
            pos.mae, pos.mfe = min(pos.mae, mtm), max(pos.mfe, mtm)
        # 공백 때문에 미뤄진 강제 청산
        if pending_forced is not None and pos is not None and np.isfinite(z):
            close_pos(m, z, L, A, pending_forced, True, "DATA_GAP_EXIT")
            pending_forced = None
        breach = np.isfinite(z) and abs(z) > e
        if breach and not in_episode:
            episode += 1
        in_episode = bool(breach)
        if ss is None:
            if breach:
                signals.append(_sig(trading_date, candidate, m, z, episode, False, "FUNDING_BLACKOUT"))
            continue
        last = m == ss.end.tz_convert("UTC")
        if not np.isfinite(z):
            if pos is not None and last and ss.end_reason != "HOLD":
                pending_forced = ss.end_reason
            continue
        # (1) 청산
        if pos is not None:
            d = pos.direction
            tgt = (d > 0 and z >= -x) or (d < 0 and z <= x)
            stp = (not tgt) and ((d > 0 and z <= -s_stop) or (d < 0 and z >= s_stop))
            if tgt or stp:
                close_pos(m, z, L, A, "TARGET" if tgt else "STOP", False)
                if stp and rearm:
                    armed = False
            elif last and ss.end_reason != "HOLD":
                close_pos(m, z, L, A, ss.end_reason, True)
        # (2) 재무장
        if abs(z) < e:
            armed = True
        # (3) 진입
        if breach:
            if last:
                signals.append(_sig(trading_date, candidate, m, z, episode, False, "AFTER_CUTOFF"))
            elif pos is not None:
                signals.append(_sig(trading_date, candidate, m, z, episode, False, "ALREADY_IN_POSITION"))
            elif not armed:
                signals.append(_sig(trading_date, candidate, m, z, episode, False, "NOT_ARMED"))
            else:
                d = 1 if z < -e else -1
                refs = {"L": L, "A": A}
                legs = []
                for key, w, side in (("A", wA, d), ("L", wL, -d)):
                    notional = C * w
                    qty = np.floor(notional / refs[key] / step) * step
                    p, basis = _fill(md, cfg, key, m, side, qty * refs[key], refs[key], False)
                    legs.append(Leg(key, side, float(qty), p, refs[key], basis))
                pos = Position(d, m, z, ss.idx, legs)
                signals.append(_sig(trading_date, candidate, m, z, episode, True, ""))
    if pos is not None:   # 거래일 끝까지 가격이 없어 못 닫음
        zl, L, A = np.nan, np.nan, np.nan
        for m in reversed(day_minutes):
            zl, L, A = zval(m)
            if np.isfinite(zl):
                break
        if np.isfinite(zl):
            close_pos(m, zl, L, A, "FORCED_0759", True, "DATA_GAP_EXIT_LAST_PRICE")
    return {"trades": trades, "signals": signals, "funding": fund_rows}


def _sig(td, cand, m, z, ep, executed, reason):
    return {"trading_date": td, "candidate": cand, "minute_utc": str(m), "z": float(z),
            "direction": "LONG_SPREAD" if z < 0 else "SHORT_SPREAD", "episode_id": ep,
            "executed": executed, "skip_reason": reason}
