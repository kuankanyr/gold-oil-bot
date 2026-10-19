#!/usr/bin/env python3
"""
regime_monitor.py - FLI-R5.6 regime monitor -> Telegram (Richie bot)

Every run (GitHub Actions, every 30 min) it checks each instrument on the last
CLOSED 30m bar against the FLI-R5.6 entry gates:

  Daily Follow Line direction (trade side)      - "With trend timeframe"
  Regime: ADX(14) >= 16, efficiency(10) >= 0.18, ATR / ATR50 <= 1.8
  Daily ADX >= 20                                - re-entries enabled
  4H Follow Line, 30m Follow Line, Andean (Bull/Bear > signal) - shown for context

FAVORABLE = regime passes AND the Daily trend points a side this instrument is
allowed to trade (SIDES below). When an instrument turns favorable, the bot sends
the full table to Telegram. State is kept in regime_state.json so you only get
one message per change, not one every 30 minutes.

Env vars:
  TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID   (required to send)
  NOTIFY_OFF=1        also message when an instrument stops being favorable
  FORCE_SEND=1        send the table now regardless of changes (test / manual run)
  SEND_ALWAYS=1       send the table on every run (hourly update), skipped while all markets are closed
  DRY_RUN=1           print instead of sending
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

# ------------------------------------------------------------------ config
# name: (yahoo ticker, allowed sides, decimals)
#   sides "both" = longs in Daily up-trend, shorts in Daily down-trend
#   sides "long" = longs only while Daily is up (backtest: oil shorts lost money)
INSTRUMENTS = {
    "XAUUSD": ("GC=F", "both", 1),
    "XAUEUR": ("GC=F/EURUSD=X", "both", 1),   # synthetic: gold / EURUSD
    "BRENT": ("BZ=F", "long", 2),
    "WTI": ("CL=F", "long", 2),
    "EURUSD": ("EURUSD=X", "both", 4),
    "USDJPY": ("USDJPY=X", "both", 2),
    # indices - longs only while Daily is up (your GER40 preset rule; US indices not yet backtested)
    "GER40": ("^GDAXI", "long", 0),    # DAX cash index: Frankfurt hours only
    "US500": ("ES=F", "long", 0),      # S&P 500 futures (~23h/day)
    "US30": ("YM=F", "long", 0),       # Dow futures
    "US100": ("NQ=F", "long", 0),      # Nasdaq 100 futures
}
STATE_FILE = os.environ.get("REGIME_STATE_FILE", "regime_state.json")

BB_LEN, BB_DEV, FL_ATR = 21, 1.0, 5
AND_LEN, AND_SIG = 50, 9
MIN_ADX, EFF_LEN, MIN_EFF = 16.0, 10, 0.18
ATR_LEN, ATR_BASE_LEN, MAX_ATR_RATIO = 14, 50, 1.8
RE_MIN_DADX = 20.0
BKK = "Asia/Bangkok"


# ------------------------------------------------------------------ data
def yf_download(ticker, period, interval):
    import yfinance as yf
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
    idx = df.index
    df.index = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    return df[df["high"] >= df["low"]]


def fetch(ticker, period, interval):
    if "/" in ticker:  # synthetic cross, e.g. gold in EUR
        a, b = ticker.split("/")
        da, db = yf_download(a, period, interval), yf_download(b, period, interval)
        if da is None or db is None:
            return None
        j = da.join(db[["close"]].rename(columns={"close": "fx"}), how="inner")
        out = j[["open", "high", "low", "close"]].div(j["fx"], axis=0)
        return out
    return yf_download(ticker, period, interval)


def drop_open_bar(df, minutes):
    """Keep only bars that have finished."""
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(minutes=minutes) <= now]


def resample_fx(df, rule):
    """Resample hourly bars to 4H / Daily with the FX day starting 17:00 New York."""
    ny = df.index.tz_convert("America/New_York")
    key = (ny + pd.Timedelta(hours=7)).tz_localize(None)
    bucket = key.normalize() if rule == "D" else key.floor(rule)
    g = df.groupby(bucket).agg(open=("open", "first"), high=("high", "max"),
                               low=("low", "min"), close=("close", "last"))
    return g


# ------------------------------------------------------------------ indicators
def rma(x, n):
    return pd.Series(x).ewm(alpha=1.0 / n, adjust=False).mean().values


def true_range(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def follow_line(df):
    h, l, c = df.high.values, df.low.values, df.close.values
    s = pd.Series(c)
    basis = s.rolling(BB_LEN).mean().values
    dev = s.rolling(BB_LEN).std(ddof=0).values * BB_DEV
    a = rma(true_range(h, l, c), FL_ATR)
    tl, tr = np.nan, 0
    out_tl, out_tr = np.full(len(c), np.nan), np.zeros(len(c), dtype=int)
    for i in range(len(c)):
        if not np.isnan(basis[i]):
            nt = tl
            if c[i] > basis[i] + dev[i]:
                cand = l[i] - a[i]
                nt = cand if np.isnan(tl) else max(cand, tl)
            elif c[i] < basis[i] - dev[i]:
                cand = h[i] + a[i]
                nt = cand if np.isnan(tl) else min(cand, tl)
            if not np.isnan(tl) and not np.isnan(nt):
                tr = 1 if nt > tl else -1 if nt < tl else tr
            tl = nt
        out_tl[i], out_tr[i] = tl, tr
    return out_tl, out_tr


def adx(df, n=14):
    h, l, c = df.high.values, df.low.values, df.close.values
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    trr = rma(true_range(h, l, c), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = 100 * rma(pdm, n) / trr
        ndi = 100 * rma(ndm, n) / trr
        dx = 100 * np.abs(pdi - ndi) / (pdi + ndi)
    return rma(np.nan_to_num(dx), n)


def andean(df):
    o, c = df.open.values, df.close.values
    a = 2.0 / (AND_LEN + 1)
    bull, bear = np.zeros(len(c)), np.zeros(len(c))
    up1 = None
    for i in range(len(c)):
        C, O = c[i], o[i]
        if up1 is None:
            up1, up2, dn1, dn2 = max(C, O), max(C * C, O * O), min(C, O), min(C * C, O * O)
        else:
            up1 = max(C, O, up1 - (up1 - C) * a)
            up2 = max(C * C, O * O, up2 - (up2 - C * C) * a)
            dn1 = min(C, O, dn1 + (C - dn1) * a)
            dn2 = min(C * C, O * O, dn2 + (C * C - dn2) * a)
        bull[i] = np.sqrt(max(dn2 - dn1 * dn1, 0.0))
        bear[i] = np.sqrt(max(up2 - up1 * up1, 0.0))
    sig = pd.Series(np.maximum(bull, bear)).ewm(span=AND_SIG, adjust=False).mean().values
    return bull[-1] > sig[-1], bear[-1] > sig[-1]


# ------------------------------------------------------------------ evaluation
def evaluate(name, m30, h1, sides):
    m30 = drop_open_bar(m30, 30)
    h1 = drop_open_bar(h1, 60)
    if len(m30) < 120 or len(h1) < 24 * 30:
        return None
    daily = resample_fx(h1, "D").iloc[:-1]        # last CLOSED FX day
    h4 = resample_fx(h1, "4h").iloc[:-1]          # last CLOSED 4H bar (approx.)
    _, d_tr = follow_line(daily)
    d_adx = adx(daily)
    _, h4_tr = follow_line(h4)
    tl, m_tr = follow_line(m30)

    h, l, c = m30.high.values, m30.low.values, m30.close.values
    atr = rma(true_range(h, l, c), ATR_LEN)
    atr_base = pd.Series(atr).rolling(ATR_BASE_LEN).mean().values
    adx30 = adx(m30)
    cs = pd.Series(c)
    net = abs(c[-1] - c[-1 - EFF_LEN])
    path = cs.diff().abs().iloc[-EFF_LEN:].sum()
    eff = net / path if path > 0 else 0.0
    atr_ratio = atr[-1] / atr_base[-1] if atr_base[-1] > 0 else 1.0
    a_bull, a_bear = andean(m30)

    adx_ok = adx30[-1] >= MIN_ADX
    eff_ok = eff >= MIN_EFF
    atr_ok = atr_ratio <= MAX_ATR_RATIO
    regime_ok = adx_ok and eff_ok and atr_ok

    daily_dir = int(d_tr[-1])
    side_ok = daily_dir == 1 or (daily_dir == -1 and sides == "both")
    favorable = regime_ok and side_ok
    reasons = []
    if not adx_ok:
        reasons.append("ADX low")
    if not eff_ok:
        reasons.append("choppy")
    if not atr_ok:
        reasons.append("ATR spike")
    if not side_ok:
        reasons.append("Daily down (longs only)" if daily_dir == -1 else "no Daily trend")
    side = "LONG" if daily_dir == 1 else "SHORT" if daily_dir == -1 else "-"
    return dict(
        name=name, favorable=bool(favorable), side=side if side_ok else "-",
        daily=daily_dir, h4=int(h4_tr[-1]), m30=int(m_tr[-1]),
        adx=float(adx30[-1]), eff=float(eff), atr_ratio=float(atr_ratio),
        d_adx=float(d_adx[-1]), re_on=bool(d_adx[-1] >= RE_MIN_DADX),
        andean="bull" if a_bull and not a_bear else "bear" if a_bear and not a_bull else "mixed",
        aligned=bool(side_ok) and bool((daily_dir == 1 and m_tr[-1] == 1 and h4_tr[-1] == 1) or
                             (daily_dir == -1 and m_tr[-1] == -1 and h4_tr[-1] == -1)),
        price=float(c[-1]), bar=m30.index[-1].isoformat(), reasons=reasons,
    )


# ------------------------------------------------------------------ telegram
ARROW = {1: "▲", -1: "▼", 0: "·"}


def build_message(rows, turned_on, turned_off):
    now = datetime.now(timezone.utc).astimezone(pd.Timestamp.now(tz=BKK).tz)
    lines = ["<b>📊 FLI regime update</b>  " + now.strftime("%a %d %b %H:%M") + " BKK"]
    if turned_on:
        lines.append("🟢 <b>Now favorable:</b> " + ", ".join(
            f"{r['name']} {r['side']}" for r in rows if r["name"] in turned_on))
    if turned_off:
        lines.append("⚪ No longer favorable: " + ", ".join(turned_off))
    tbl = ["Inst    Side  D 4H 30 ADX Eff ATRx DADX",
           "------  ----- - -- -- --- --- ---- ----"]
    for r in rows:
        tbl.append(f"{r['name']:<6}  {r['side']:<5} {ARROW[r['daily']]}  {ARROW[r['h4']]}  {ARROW[r['m30']]} "
                   f"{r['adx']:>3.0f} {r['eff']:.2f} {r['atr_ratio']:>4.1f} {r['d_adx']:>4.0f}")
    lines.append("<pre>" + "\n".join(tbl) + "</pre>")
    notes = []
    for r in rows:
        if r["favorable"]:
            extra = "re-entries ON" if r["re_on"] else "signals only (Daily ADX &lt; 20)"
            align = ", 4H+30m aligned" if r["aligned"] else ""
            notes.append(f"🟢 <b>{r['name']}</b> {r['side']} — {extra}{align}, Andean {r['andean']}")
        else:
            notes.append(f"🔴 {r['name']} — " + ", ".join(r["reasons"]))
    lines.append("\n".join(notes))
    lines.append("<i>Gates: ADX≥16 · Eff≥0.18 · ATRx≤1.8 · trade with Daily. Regime only, not an entry signal.</i>")
    return "\n\n".join(lines)


def send(text):
    if os.environ.get("DRY_RUN"):
        print(text)
        return
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat:
        sys.exit("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML",
                            "disable_web_page_preview": True}, timeout=30)
    if not r.ok:
        sys.exit(f"Telegram error {r.status_code}: {r.text}")


# ------------------------------------------------------------------ main
def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def main(fetcher=fetch):
    state = load_state()
    prev_fav = state.get("favorable", {})
    rows = []
    for name, (ticker, sides, _) in INSTRUMENTS.items():
        try:
            m30 = fetcher(ticker, "59d", "30m")
            h1 = fetcher(ticker, "400d", "1h")
            r = evaluate(name, m30, h1, sides) if m30 is not None and h1 is not None else None
        except Exception as e:  # one bad feed shouldn't stop the others
            print(f"{name}: {e}")
            r = None
        if r is None:
            print(f"{name}: no data")
            continue
        # market closed / stale feed: keep previous state
        age_h = (pd.Timestamp.now(tz="UTC") - pd.Timestamp(r["bar"])).total_seconds() / 3600
        r["stale"] = age_h > 3
        rows.append(r)

    if not rows:
        sys.exit("No instruments evaluated.")
    live = [r for r in rows if not r["stale"]]
    turned_on = [r["name"] for r in live if r["favorable"] and not prev_fav.get(r["name"], False)]
    turned_off = [r["name"] for r in live if not r["favorable"] and prev_fav.get(r["name"], False)]

    for r in rows:
        print(f"{r['name']:<7} fav={r['favorable']!s:<5} side={r['side']:<5} D={r['daily']:+d} "
              f"ADX={r['adx']:.0f} eff={r['eff']:.2f} atrx={r['atr_ratio']:.2f} dADX={r['d_adx']:.0f} "
              f"{'STALE' if r['stale'] else ''} {r['reasons']}")

    notify = bool(turned_on) or (os.environ.get("NOTIFY_OFF") == "1" and bool(turned_off))
    always = os.environ.get("SEND_ALWAYS") == "1" and bool(live)   # no messages over the weekend
    if os.environ.get("FORCE_SEND") == "1" or notify or always:
        send(build_message(rows, turned_on, turned_off))
        print("sent")
    else:
        print("no change -> no message")

    new_fav = dict(prev_fav)
    for r in live:
        new_fav[r["name"]] = bool(r["favorable"])
    with open(STATE_FILE, "w") as f:
        json.dump({"favorable": new_fav, "updated": datetime.now(timezone.utc).isoformat()}, f, indent=2)


if __name__ == "__main__":
    main()
