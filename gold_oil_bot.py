#!/usr/bin/env python3
"""
Gold & Oil briefing -> Telegram
Covers XAUUSD, XAUEUR, WTI and Brent: price snapshot + trend, the headlines
most likely to move them, and (optional) a short AI-written bias read.

Env vars:
  TELEGRAM_BOT_TOKEN   required (from @BotFather)
  TELEGRAM_CHAT_ID     required (your chat id, or a channel/group id)
  ANTHROPIC_API_KEY    optional - enables the per-headline bullish/bearish read
  ANTHROPIC_MODEL      optional - default claude-sonnet-5-5
  LOOKBACK_HOURS       optional - news window, default 12 (auto-extends on Monday mornings)
  TZ_NAME              optional - display timezone, default Asia/Bangkok
  STATE_FILE           optional - remembers sent headlines between runs, default .seen.json

Usage:
  python gold_oil_bot.py            # build and send
  python gold_oil_bot.py --dry-run  # print to terminal instead of sending
"""
import argparse
import datetime as dt
import html
import json
import math
import os
import re
import sys
import time
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import feedparser
import requests

# ----------------------------------------------------------------------------- config

TZ = ZoneInfo(os.getenv("TZ_NAME", "Asia/Bangkok"))
UA = "Mozilla/5.0 (compatible; gold-oil-briefing/1.0)"
MAX_PER_CLASS = 4          # max headlines per asset class (gold / oil / macro)
GOLD, OIL = ["XAUUSD", "XAUEUR"], ["WTI", "BRENT"]
ALL = GOLD + OIL

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}+when:{d}d&hl=en-US&gl=US&ceid=US:en"
NEWS_QUERIES = {
    "gold": ["gold price", "XAUUSD", "gold futures", "central bank gold buying"],
    "oil": ["crude oil price", "Brent crude", "WTI crude", "OPEC+"],
    "macro": ["Federal Reserve rates", "US dollar index", "Treasury yields", "ECB interest rates"],
}
EXTRA_FEEDS = [
    "https://oilprice.com/rss/main",
]

# regex -> (tag shown in message, assets affected, weight)
DRIVERS = [
    (r"\bfed\b|fomc|powell|rate cut|rate hike|interest rate", "Fed/rates", ALL, 3),
    (r"\bcpi\b|inflation|\bpce\b|\bppi\b", "Inflation", ALL, 3),
    (r"payroll|jobs report|unemployment|jobless", "US jobs", ALL, 3),
    (r"dollar|\bdxy\b|greenback", "USD", GOLD + OIL, 2),
    (r"yield|treasur", "Yields", GOLD, 2),
    (r"\becb\b|lagarde|euro\b|eurozone", "ECB/EUR", ["XAUEUR"], 2),
    (r"opec", "OPEC+", OIL, 3),
    (r"inventor|stockpile|\beia\b|cushing|crude draw|crude build", "Inventories", OIL, 3),
    (r"sanction|iran|russia|israel|middle east|houthi|red sea|hormuz|ukraine|\bwar\b|missile|attack|ceasefire|strike",
     "Geopolitics", ALL, 3),
    (r"tariff|trade war|recession|\bpmi\b|\bgdp\b|china demand|stimulus", "Growth/demand", ALL, 2),
    (r"central bank.*(buy|purchas)|gold reserve|pboc|\betf\b", "Gold flows", GOLD, 2),
    (r"safe.haven|risk.off|risk aversion", "Safe-haven", GOLD, 2),
    (r"refiner|hurricane|outage|pipeline|production cut|output|shale|rig count", "Supply", OIL, 2),
]
DIRECT = [
    (r"gold|bullion|xau", GOLD),
    (r"\boil\b|crude|brent|\bwti\b", OIL),
]
# retail / local-price noise that rarely matters for XAU or crude benchmarks
NOISE = re.compile(r"price today|rate today|rates today|in india|jewell?ery|karat|carat|petrol price|"
                   r"diesel price|gold loan|\bmcx\b|sovereign gold bond|silver price today", re.I)


# ----------------------------------------------------------------------------- prices

def fetch_prices():
    """Daily closes -> last price, 1d/5d change, trend vs SMA20/50, RSI14."""
    import yfinance as yf

    tickers = ["GC=F", "CL=F", "BZ=F", "EURUSD=X"]
    close = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=False)["Close"].ffill()
    series = {
        "XAUUSD": close["GC=F"],                       # COMEX gold futures as spot proxy
        "XAUEUR": close["GC=F"] / close["EURUSD=X"],
        "WTI": close["CL=F"],
        "BRENT": close["BZ=F"],
    }
    out = {}
    for name, s in series.items():
        s = s.dropna()
        if len(s) < 55:
            out[name] = None
            continue
        last = float(s.iloc[-1])
        sma20, sma50 = float(s.rolling(20).mean().iloc[-1]), float(s.rolling(50).mean().iloc[-1])
        delta = s.diff()
        gain = delta.clip(lower=0).rolling(14).mean().iloc[-1]
        loss = (-delta.clip(upper=0)).rolling(14).mean().iloc[-1]
        rsi = 100.0 if loss == 0 else 100 - 100 / (1 + gain / loss)
        if last > sma20 > sma50:
            trend = "Up"
        elif last < sma20 < sma50:
            trend = "Down"
        else:
            trend = "Mixed"
        out[name] = {
            "last": last,
            "chg1d": (last / float(s.iloc[-2]) - 1) * 100,
            "chg5d": (last / float(s.iloc[-6]) - 1) * 100,
            "sma20": sma20,
            "sma50": sma50,
            "rsi": float(rsi),
            "trend": trend,
        }
    if out.get("WTI") and out.get("BRENT"):
        out["_spread"] = out["BRENT"]["last"] - out["WTI"]["last"]
    return out


# ----------------------------------------------------------------------------- news

def _get_feed(url):
    r = requests.get(url, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    return feedparser.parse(r.content)


def _clean_title(title, source):
    title = html.unescape(title or "").strip()
    if source and title.endswith(" - " + source):
        title = title[: -len(source) - 3]
    return title


def score_headline(title):
    t = title.lower()
    score, tags, assets = 0, [], set()
    for pattern, tag, affected, weight in DRIVERS:
        if re.search(pattern, t):
            score += weight
            tags.append(tag)
            assets.update(affected)
    direct = set()
    for pattern, affected in DIRECT:
        if re.search(pattern, t):
            direct.update(affected)
    if direct:
        score += 1
        # a gold/oil headline only affects that class, even if drivers are broad
        assets = (assets & direct) or direct
    return score, tags, assets, bool(direct)


def fetch_news(lookback_h):
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=lookback_h)
    days = max(1, math.ceil(lookback_h / 24))
    urls = [GOOGLE_NEWS.format(q=quote_plus(q), d=days) for qs in NEWS_QUERIES.values() for q in qs] + EXTRA_FEEDS

    seen_keys, items = set(), []
    for url in urls:
        try:
            feed = _get_feed(url)
        except Exception as e:
            print(f"[warn] feed failed: {url} ({e})", file=sys.stderr)
            continue
        for e in feed.entries:
            ts = e.get("published_parsed") or e.get("updated_parsed")
            if not ts:
                continue
            published = dt.datetime.fromtimestamp(time.mktime(ts), dt.timezone.utc)
            if published < cutoff:
                continue
            source = (e.get("source") or {}).get("title") or feed.feed.get("title", "")
            title = _clean_title(e.get("title"), source)
            if not title or NOISE.search(title):
                continue
            key = re.sub(r"\W+", " ", title.lower()).strip()[:80]
            if key in seen_keys:
                continue
            seen_keys.add(key)

            score, tags, assets, direct = score_headline(title)
            if score < 2 or not assets:
                continue
            items.append({
                "title": title, "link": e.get("link", ""), "source": source,
                "published": published, "score": score, "tags": tags, "assets": assets,
                "cls": "gold" if assets <= set(GOLD) else "oil" if assets <= set(OIL) else "macro",
                "key": key,
            })

    items.sort(key=lambda x: (x["score"], x["published"]), reverse=True)
    picked, counts = [], {"gold": 0, "oil": 0, "macro": 0}
    for it in items:
        if counts[it["cls"]] < MAX_PER_CLASS:
            picked.append(it)
            counts[it["cls"]] += 1
    return picked


# ----------------------------------------------------------------------------- state (avoid repeats)

def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_state(path, state, news):
    now = time.time()
    state.update({it["key"]: now for it in news})
    state = {k: v for k, v in state.items() if now - v < 4 * 86400}  # keep 4 days
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[warn] could not save state: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------- AI read per headline (optional)

def llm_reads(prices, news, lookback_h):
    """Returns ({headline_no: {"bias","asset","why"}}, {"gold": str, "oil": str}) or ({}, {})."""
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key or not news:
        return {}, {}
    price_lines = "\n".join(
        f"{k}: {v['last']:.2f} | 1d {v['chg1d']:+.2f}% | 5d {v['chg5d']:+.2f}% | "
        f"SMA20 {v['sma20']:.2f} | SMA50 {v['sma50']:.2f} | RSI14 {v['rsi']:.0f} | trend {v['trend']}"
        for k, v in prices.items() if not k.startswith("_") and v
    ) or "(price data unavailable)"
    headline_lines = "\n".join(
        f"[{i}] ({it['source']}) {it['title']}  -- relevant to: {', '.join(sorted(it['assets']))}"
        for i, it in enumerate(news, 1)
    )
    prompt = f"""You are a commodities analyst. For each headline, judge its likely price impact on gold (XAUUSD/XAUEUR) and/or crude oil (WTI/Brent).

Market snapshot (daily data):
{price_lines}

Headlines from the last {lookback_h} hours:
{headline_lines}

Return ONLY JSON, no other text, in this shape:
{{"items": [{{"i": 1, "asset": "Gold" | "Oil" | "Gold & Oil" | "XAUEUR", "bias": "Bullish" | "Bearish" | "Mixed" | "Neutral", "why": "one sentence, max 25 words, the mechanism (e.g. lower rate expectations weaken USD and support non-yielding gold)"}}],
 "overall": {{"gold": "Bullish|Bearish|Neutral - max 15 words", "oil": "Bullish|Bearish|Neutral - max 15 words"}}}}

Rules: one item per headline number; reason only from the headline text and numbers above; if a headline's impact is unclear, use "Neutral" and say why; never invent facts."""
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={
                "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
                "max_tokens": 2000,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=90,
        )
        r.raise_for_status()
        text = "".join(b.get("text", "") for b in r.json()["content"])
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        reads = {int(x["i"]): x for x in data.get("items", []) if "i" in x}
        return reads, data.get("overall") or {}
    except Exception as e:
        print(f"[warn] analysis failed: {e}", file=sys.stderr)
        return {}, {}


# ----------------------------------------------------------------------------- message

BIAS_ICON = {"bullish": "🟢", "bearish": "🔴", "mixed": "🟡", "neutral": "⚪"}
ARROW = {"Up": "▲", "Down": "▼", "Mixed": "◆"}
NUMS = ["1️⃣", "2️⃣", "3️⃣", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]


def _ago(published):
    mins = int((dt.datetime.now(dt.timezone.utc) - published).total_seconds() // 60)
    return f"{mins}m ago" if mins < 60 else f"{mins // 60}h ago"


def _price_line(prices, assets):
    parts = []
    for name in ["XAUUSD", "XAUEUR", "WTI", "BRENT"]:
        if name in assets:
            p = prices.get(name)
            parts.append(f"{name} {p['last']:,.2f} ({p['chg1d']:+.2f}%) {ARROW[p['trend']]}" if p else f"{name} n/a")
    return " · ".join(parts)


def build_message(prices, news, reads, overall, lookback_h):
    now = dt.datetime.now(TZ)
    out = [f"🟡🛢 <b>Gold &amp; Oil News</b> — <i>{now:%a %d %b, %H:%M}</i>", ""]

    if not news:
        out.append(f"No new market-moving headlines in the last {lookback_h}h.")
        out.append("💰 " + html.escape(_price_line(prices, ALL)))
        return "\n".join(out)

    for i, it in enumerate(news, 1):
        num = NUMS[i - 1] if i <= len(NUMS) else f"{i}."
        link = html.escape(it["link"], quote=True)
        out.append(f"{num} <b>{html.escape(it['title'])}</b>")
        out.append(f"🗞 <a href=\"{link}\">{html.escape(it['source'] or 'Source')}</a> · {_ago(it['published'])}")
        out.append("💰 " + html.escape(_price_line(prices, it["assets"])))
        rd = reads.get(i)
        if rd:
            bias = str(rd.get("bias", "Neutral"))
            icon = BIAS_ICON.get(bias.lower(), "⚪")
            out.append(f"🧭 {icon} <b>{html.escape(bias)} {html.escape(str(rd.get('asset', '')))}</b> — "
                       f"{html.escape(str(rd.get('why', '')))}")
        else:
            out.append(f"🧭 Driver: {html.escape(' · '.join(it['tags']) or 'Direct price news')}")
        out.append("")

    if overall:
        out.append("📊 <b>Overall</b>")
        if overall.get("gold"):
            out.append(f"Gold: {html.escape(str(overall['gold']))}")
        if overall.get("oil"):
            out.append(f"Oil: {html.escape(str(overall['oil']))}")
    if "_spread" in prices:
        out.append(f"<i>Brent–WTI spread {prices['_spread']:.2f} · trend ▲▼◆ = price vs 20/50-day avg</i>")
    return "\n".join(out).rstrip()


def split_message(text, limit=4000):
    """Split on blank lines so a headline block is never cut in half."""
    chunks, cur = [], ""
    for block in text.split("\n\n"):
        if len(cur) + len(block) + 2 > limit and cur:
            chunks.append(cur.rstrip())
            cur = ""
        cur += block + "\n\n"
    if cur.strip():
        chunks.append(cur.rstrip())
    return chunks


def send_telegram(text):
    token, chat_id = os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        sys.exit("TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID must be set (or use --dry-run).")
    for chunk in split_message(text):
        r = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": chunk, "parse_mode": "HTML",
                  "link_preview_options": {"is_disabled": True}},
            timeout=20,
        )
        if not r.ok:
            raise RuntimeError(f"Telegram error {r.status_code}: {r.text}")


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LOOKBACK_HOURS", "12")))
    args = ap.parse_args()

    lookback = args.lookback
    now_utc = dt.datetime.now(dt.timezone.utc)
    if now_utc.weekday() == 0 and now_utc.hour < 6:   # Monday morning: cover the weekend
        lookback = max(lookback, 60)

    try:
        prices = fetch_prices()
    except Exception as e:
        print(f"[warn] prices failed: {e}", file=sys.stderr)
        prices = {}

    state_path = os.getenv("STATE_FILE", ".seen.json")
    state = load_state(state_path)
    news = [it for it in fetch_news(lookback) if it["key"] not in state]

    reads, overall = llm_reads(prices, news, lookback)
    msg = build_message(prices, news, reads, overall, lookback)

    if args.dry_run:
        print(msg)
    else:
        send_telegram(msg)
        save_state(state_path, state, news)
        print(f"Sent: {len(news)} headlines, reads={len(reads)}")


if __name__ == "__main__":
    main()
