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
  SKIP_EMPTY           optional - set to 1 to send nothing when there are no new headlines
  CAL_CURRENCIES       optional - calendar currencies, default USD,EUR,GBP,JPY,CNY,CAD,AUD,CHF
  CAL_DAILY_HOUR       optional - local hour to send the daily calendar, default 7
  ALERT_WINDOW_MIN     optional - minutes before a High-impact event to alert, default 90
  WATCH_MINUTES        optional - same as --watch: poll every minute for N minutes, send bullet alerts

Usage:
  python gold_oil_bot.py            # build and send
  python gold_oil_bot.py --dry-run  # print to terminal instead of sending
  python gold_oil_bot.py --watch 600   # real-time mode: bullet alerts every minute for 10 hours
"""
import argparse
import calendar
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

GOOGLE_NEWS = "https://news.google.com/rss/search?q={q}+when:{d}&hl=en-US&gl=US&ceid=US:en"
NEWS_QUERIES = {
    "gold": ["gold price", "XAUUSD", "gold futures", "central bank gold buying"],
    "oil": ["crude oil price", "Brent crude", "WTI crude", "OPEC+"],
    "macro": ["Federal Reserve rates", "US dollar index", "Treasury yields", "ECB interest rates"],
}
# fast feeds: polled every minute in --watch mode (Google News is slower and polled less often)
EXTRA_FEEDS = [
    "https://www.fxstreet.com/rss/news",                 # FX / gold / oil wires, minutes after events
    "https://www.investing.com/rss/news_11.rss",         # Investing.com commodities news
    "https://oilprice.com/rss/main",
]
TRUMP_FEED = "https://trumpstruth.org/feed"              # Trump's Truth Social posts (unofficial mirror)
TRUMP_MARKET = re.compile(
    r"tariff|trade|china|xi\b|fed\b|federal reserve|powell|interest rate|rates|inflation|dollar|gold|oil|"
    r"opec|drill|energy|gas price|iran|israel|russia|putin|ukraine|venezuela|sanction|war\b|strike|missile|"
    r"ceasefire|peace|deal|embargo|blockade|hormuz|stock market|recession|treasur|bond|debt|budget|shutdown", re.I)

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
    (r"trump|white house", "Trump", ALL, 2),
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

def _live_prices(yf, tickers):
    """Latest intraday price per ticker (15-min bars), so 'last' is live, not yesterday's close."""
    try:
        intra = yf.download(tickers, period="5d", interval="15m", progress=False, auto_adjust=False)["Close"]
        return {t: float(intra[t].dropna().iloc[-1]) for t in tickers if t in intra and intra[t].notna().any()}
    except Exception as e:
        print(f"[warn] intraday prices failed: {e}", file=sys.stderr)
        return {}


def fetch_prices():
    """Live last price, change vs previous daily close, 5d change, trend vs SMA20/50, RSI14."""
    import yfinance as yf

    tickers = ["GC=F", "CL=F", "BZ=F", "EURUSD=X"]
    daily = yf.download(tickers, period="6mo", interval="1d", progress=False, auto_adjust=False)["Close"]
    live = _live_prices(yf, tickers)
    today = dt.datetime.now(dt.timezone.utc).date()

    def clean(t):
        # each ticker on its own calendar - never forward-fill one ticker onto another's dates
        s = daily[t].dropna().copy()
        if t in live:
            if len(s) and s.index[-1].date() >= today:
                s.iloc[-1] = live[t]                     # update today's bar with the live price
            else:
                s.loc[s.index[-1] + dt.timedelta(days=1) if len(s) else dt.datetime.now()] = live[t]
        return s

    gold, eur = clean("GC=F"), clean("EURUSD=X")
    xaueur = (gold / eur.reindex(gold.index, method="ffill")).dropna()   # EUR only filled onto gold's dates
    series = {"XAUUSD": gold, "XAUEUR": xaueur, "WTI": clean("CL=F"), "BRENT": clean("BZ=F")}

    out = {}
    for name, s in series.items():
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


_STOP = set("a an the as and or of on in to for at by with from is are amid after over its".split())


def _words(title):
    return {w for w in re.findall(r"[a-z0-9]+", title.lower()) if w not in _STOP and len(w) > 2}


def _similar(a, b, threshold=0.5):
    """Same story? Share of the shorter headline's key words found in the other."""
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / min(len(wa), len(wb)) >= threshold


_UP = set("rise rises rising gain gains jump jumps climb climbs rally rallies surge surges soar soars rebound rebounds advance advances".split())
_DOWN = set("fall falls falling drop drops dip dips slide slides slump slumps sink sinks tumble tumbles plunge plunges "
            "decline declines retreat retreats slip slips ease eases".split())


def _direction(title):
    w = set(re.findall(r"[a-z]+", title.lower()))
    up, down = bool(w & _UP), bool(w & _DOWN)
    return "up" if up and not down else "down" if down and not up else None


def _same_story(x, y):
    dx, dy = _direction(x["title"]), _direction(y["title"])
    if dx and dy and dx != dy:
        return False                                   # "oil rises" vs "oil dips" are different news
    if _similar(x["title"], y["title"], 0.6):
        return True
    # same asset, same drivers, and a decent word overlap (e.g. three "gold falls on dollar & yields" pieces)
    return x["cls"] == y["cls"] and x["tags"] and x["tags"] == y["tags"] and _similar(x["title"], y["title"], 0.35)


def _entry_time(e):
    ts = e.get("published_parsed") or e.get("updated_parsed")
    if ts:
        return dt.datetime.fromtimestamp(calendar.timegm(ts), dt.timezone.utc)
    raw = e.get("published") or e.get("updated")
    if raw:                                                  # e.g. Investing.com "2026-10-07 20:36:35"
        try:
            t = dt.datetime.fromisoformat(raw.strip().replace("Z", "+00:00"))
            return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            pass
    return None


def _trump_items(cutoff):
    out = []
    try:
        feed = _get_feed(TRUMP_FEED)
    except Exception as e:
        print(f"[warn] trump feed failed: {e}", file=sys.stderr)
        return out
    for e in feed.entries:
        published = _entry_time(e)
        if not published or published < cutoff:
            continue
        text = re.sub(r"<[^>]+>", " ", html.unescape(e.get("description") or e.get("summary") or ""))
        text = re.sub(r"\s+", " ", text).strip()
        if not text or not TRUMP_MARKET.search(text):
            continue                                         # skip posts with nothing market-related
        score, tags, assets, _ = score_headline(text)
        short = text if len(text) <= 220 else text[:217].rsplit(" ", 1)[0] + "…"
        out.append({
            "title": short, "link": e.get("link", ""), "source": "Truth Social",
            "published": published, "score": score + 4, "tags": ["Trump"] + [t for t in tags if t != "Trump"],
            "assets": assets or set(ALL), "cls": "macro", "trump": True,
            "key": "trump:" + (e.get("guid") or e.get("link") or short[:60]),
        })
    return out


def fetch_news(lookback_h, include_google=True):
    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=lookback_h)
    window = f"{max(1, math.ceil(lookback_h))}h" if lookback_h < 24 else f"{math.ceil(lookback_h / 24)}d"
    urls = list(EXTRA_FEEDS)
    if include_google:
        urls += [GOOGLE_NEWS.format(q=quote_plus(q), d=window) for qs in NEWS_QUERIES.values() for q in qs]

    seen_keys, items = set(), _trump_items(cutoff)
    for url in urls:
        try:
            feed = _get_feed(url)
        except Exception as e:
            print(f"[warn] feed failed: {url} ({e})", file=sys.stderr)
            continue
        for e in feed.entries:
            published = _entry_time(e)
            if not published or published < cutoff or published > now + dt.timedelta(minutes=10):
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
        if any(_same_story(it, p) for p in picked):
            continue                                   # same story from another outlet
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


def save_state(path, state, news, extra_keys=()):
    now = time.time()
    state.update({it["key"]: now for it in news})
    state.update({k: now for k in extra_keys})
    state = {k: v for k, v in state.items() if now - v < 4 * 86400}  # keep 4 days
    try:
        with open(path, "w") as f:
            json.dump(state, f)
    except Exception as e:
        print(f"[warn] could not save state: {e}", file=sys.stderr)


# ----------------------------------------------------------------------------- AI read per headline (optional)

def call_claude(prompt, max_tokens):
    r = requests.post(
        "https://api.anthropic.com/v1/messages",
        headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json={
            "model": os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=90,
    )
    r.raise_for_status()
    return "".join(b.get("text", "") for b in r.json()["content"]).strip()


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
        text = call_claude(prompt, 2000)
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
    if mins < 1:
        return "just now"
    return f"{mins}m ago" if mins < 60 else f"{mins // 60}h ago"


def _price_line(prices, assets):
    parts = []
    for name in ["XAUUSD", "XAUEUR", "WTI", "BRENT"]:
        if name in assets:
            p = prices.get(name)
            parts.append(f"{name} {p['last']:,.2f} ({p['chg1d']:+.2f}%) {ARROW[p['trend']]}" if p else f"{name} n/a")
    return " · ".join(parts)


def _snapshot(prices):
    def one(name, label=None):
        p = prices.get(name)
        if not p:
            return f"{label or name} n/a"
        return f"{label or name} <b>{p['last']:,.2f}</b> ({p['chg1d']:+.2f}%) {ARROW[p['trend']]}"
    gold = f"🟡 {one('XAUUSD')} · {one('XAUEUR')}"
    oil = f"🛢 {one('WTI')} · {one('BRENT', 'Brent')}"
    if "_spread" in prices:
        oil += f" · spread {prices['_spread']:.2f}"
    return gold + "\n" + oil


def build_message(prices, news, reads, overall, lookback_h):
    now = dt.datetime.now(TZ)
    out = [f"📰 <b>Gold &amp; Oil News</b> — <i>{now:%a %d %b, %H:%M}</i>", _snapshot(prices), ""]

    if not news:
        out.append(f"No new market-moving headlines in the last {lookback_h}h.")
        return "\n".join(out)

    for i, it in enumerate(news, 1):
        num = NUMS[i - 1] if i <= len(NUMS) else f"{i}."
        link = html.escape(it["link"], quote=True)
        out.append(f"{num} <b>{html.escape(it['title'])}</b>")
        out.append(f"🗞 <a href=\"{link}\">{html.escape(it['source'] or 'Source')}</a> · {_ago(it['published'])}")
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
    out.append("<i>▲▼◆ = price above / below / between its 20- &amp; 50-day averages</i>")
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


# ----------------------------------------------------------------------------- economic calendar

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"   # ForexFactory weekly feed
CAL_CURRENCIES = [c.strip() for c in os.getenv("CAL_CURRENCIES", "USD,EUR,GBP,JPY,CNY,CAD,AUD,CHF").split(",")]
ALERT_WINDOW_MIN = int(os.getenv("ALERT_WINDOW_MIN", "90"))   # heads-up when a High event is this close

FLAGS = {"USD": "🇺🇸", "EUR": "🇪🇺", "GBP": "🇬🇧", "JPY": "🇯🇵", "CNY": "🇨🇳", "CAD": "🇨🇦",
         "AUD": "🇦🇺", "CHF": "🇨🇭", "NZD": "🇳🇿"}
IMPACT_ICON = {"High": "🔴", "Medium": "🟠", "Low": "⚪"}

# what a STRONGER-than-expected result / hawkish outcome does
STRONG_EFFECT = {
    "USD": "USD↑ · Gold↓ · EUR/USD↓ · USD/JPY↑",
    "EUR": "EUR↑ · EUR/USD↑ · XAUEUR↓",
    "GBP": "GBP↑ · GBP/USD↑",
    "JPY": "JPY↑ · USD/JPY↓",
    "CNY": "China demand↑ → Oil↑ · AUD↑",
    "CAD": "CAD↑ · USD/CAD↓",
    "AUD": "AUD↑ · AUD/USD↑",
    "CHF": "CHF↑ · USD/CHF↓",
    "NZD": "NZD↑ · NZD/USD↑",
}

# (pattern, kind). kind: normal = higher is stronger, inverse = higher is weaker
EVENT_RULES = [
    (r"crude oil inventor|crude inventor", "oil"),
    (r"unemployment|jobless|claimant", "inverse"),
    (r"rate decision|rate statement|cash rate|bank rate|funds rate|policy rate|refinancing|"
     r"monetary policy|fomc|press conference|speaks|minutes|testif", "policy"),
    (r"cpi|pce|ppi|inflation|price index", "normal"),
    (r"non-farm|nonfarm|employment change|payroll|adp", "normal"),
    (r"gdp|retail sales|pmi|ism|sentiment|confidence|industrial production|durable|jolts|job openings|"
     r"trade balance|housing|home sales", "normal"),
]


def _flip(effect):
    return effect.translate(str.maketrans("↑↓", "↓↑"))


def event_playbook(title, ccy):
    t = title.lower()
    kind = next((k for p, k in EVENT_RULES if re.search(p, t)), "normal")
    strong = STRONG_EFFECT.get(ccy, f"{ccy}↑")
    if kind == "oil":
        return ["Draw (below forecast) → Oil↑ · CAD↑", "Build (above forecast) → Oil↓ · CAD↓"]
    if kind == "policy":
        return [f"Hawkish → {strong}", f"Dovish → {_flip(strong)}"]
    if kind == "inverse":
        return [f"Higher than forecast → {_flip(strong)}", f"Lower → {strong}"]
    return [f"Above forecast → {strong}", f"Below → {_flip(strong)}"]


def fetch_calendar():
    r = requests.get(CALENDAR_URL, headers={"User-Agent": UA}, timeout=20)
    r.raise_for_status()
    events = []
    for e in r.json():
        ccy, impact = e.get("country", ""), e.get("impact", "")
        title = e.get("title", "")
        if ccy not in CAL_CURRENCIES:
            continue
        is_oil = bool(re.search(r"crude oil inventor", title, re.I))
        # all High events, plus Medium for USD/EUR (drive gold & XAUEUR) and the EIA oil report
        if not (impact == "High" or is_oil or (impact == "Medium" and ccy in ("USD", "EUR"))):
            continue
        try:
            when = dt.datetime.fromisoformat(e["date"]).astimezone(dt.timezone.utc)
        except Exception:
            continue
        events.append({"title": title, "ccy": ccy, "impact": impact, "when": when,
                       "forecast": e.get("forecast") or "", "previous": e.get("previous") or ""})
    events.sort(key=lambda x: x["when"])
    return events


def _event_key(ev):
    return f"cal:{ev['ccy']}:{ev['title']}:{ev['when']:%Y%m%d%H%M}"


def _event_block(ev, show_day=False):
    local = ev["when"].astimezone(TZ)
    when = local.strftime("%a %H:%M" if show_day else "%H:%M")
    lines = [f"<b>{when}</b> {FLAGS.get(ev['ccy'], '')} {ev['ccy']} · <b>{html.escape(ev['title'])}</b> "
             f"{IMPACT_ICON.get(ev['impact'], '')}"]
    nums = " | ".join(x for x in [f"Forecast {ev['forecast']}" if ev["forecast"] else "",
                                  f"Prev {ev['previous']}" if ev["previous"] else ""] if x)
    if nums:
        lines.append(f"   {html.escape(nums)}")
    up, down = event_playbook(ev["title"], ev["ccy"])
    lines.append(f"   ↗ {html.escape(up)}")
    lines.append(f"   ↘ {html.escape(down)}")
    return "\n".join(lines)


def calendar_focus(events, prices):
    """Optional 2-3 sentence 'what matters most today' from Claude."""
    if not os.getenv("ANTHROPIC_API_KEY") or not events:
        return None
    ev_lines = "\n".join(f"{e['when'].astimezone(TZ):%H:%M} {e['ccy']} {e['title']} ({e['impact']}) "
                         f"forecast={e['forecast'] or 'n/a'} prev={e['previous'] or 'n/a'}" for e in events)
    px = ", ".join(f"{k} {v['last']:.2f} ({v['chg5d']:+.1f}% 5d, trend {v['trend']})"
                   for k, v in prices.items() if not k.startswith("_") and v) or "n/a"
    prompt = f"""Today's economic calendar (Bangkok time):
{ev_lines}

Current prices: {px}

In 2-3 plain-text sentences (no markdown, max 70 words), say which 1-2 events matter most today for gold, oil and the major FX pairs, and what result would be a surprise given forecast vs previous. Use only the data above; do not invent numbers."""
    try:
        return call_claude(prompt, 400)
    except Exception as e:
        print(f"[warn] calendar focus failed: {e}", file=sys.stderr)
        return None


def build_daily_calendar(events, prices, focus):
    now = dt.datetime.now(TZ)
    today = [e for e in events if e["when"].astimezone(TZ).date() == now.date()]
    later = [e for e in events if e["when"].astimezone(TZ).date() > now.date() and e["impact"] == "High"]
    out = [f"📅 <b>Economic Calendar</b> — <i>{now:%a %d %b} (Bangkok time)</i>", ""]
    if focus:
        out += [f"🎯 {html.escape(focus)}", ""]
    if today:
        out.append("<b>Today</b>")
        out += [_event_block(e) + "\n" for e in today]
    else:
        out += ["No major events today.", ""]
    if later:
        out.append("<b>Later this week</b> 🔴")
        for e in later[:10]:
            l = e["when"].astimezone(TZ)
            out.append(f"{l:%a %H:%M} {FLAGS.get(e['ccy'], '')} {e['ccy']} · {html.escape(e['title'])}")
    out.append("\n<i>↗/↘ = typical reaction if the result beats/misses forecast. Actual moves depend on positioning.</i>")
    return "\n".join(out)


def build_alerts(upcoming):
    out = ["⏰ <b>Coming up</b>", ""]
    for e in upcoming:
        mins = int((e["when"] - dt.datetime.now(dt.timezone.utc)).total_seconds() // 60)
        out.append(_event_block(e) + f"\n   <i>in ~{mins} min</i>\n")
    return "\n".join(out).rstrip()


def run_calendar(prices, state, force_daily=False):
    """Returns list of (message, keys_to_remember)."""
    try:
        events = fetch_calendar()
    except Exception as e:
        print(f"[warn] calendar failed: {e}", file=sys.stderr)
        return []
    msgs = []
    now_local = dt.datetime.now(TZ)
    daily_key = f"cal-daily:{now_local:%Y-%m-%d}"
    if force_daily or (daily_key not in state and now_local.hour >= int(os.getenv("CAL_DAILY_HOUR", "7"))):
        focus = calendar_focus([e for e in events if e["when"].astimezone(TZ).date() == now_local.date()], prices)
        msgs.append((build_daily_calendar(events, prices, focus), [daily_key]))
    now_utc = dt.datetime.now(dt.timezone.utc)
    upcoming = [e for e in events if e["impact"] == "High"
                and 0 < (e["when"] - now_utc).total_seconds() <= ALERT_WINDOW_MIN * 60
                and _event_key(e) not in state]
    if upcoming:
        msgs.append((build_alerts(upcoming), [_event_key(e) for e in upcoming]))
    return msgs


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


# ----------------------------------------------------------------------------- flash alerts (--watch)

def _asset_icon(it):
    if it.get("trump"):
        return "🚨"
    return {"gold": "🟡", "oil": "🛢"}.get(it["cls"], "🌐")


def flash_reads(items):
    """{index: 'short bias'} - one quick Claude call per batch, optional."""
    if not os.getenv("ANTHROPIC_API_KEY") or not items:
        return {}
    lines = "\n".join(f"[{i}] {it['title']}" for i, it in enumerate(items, 1))
    prompt = f"""Breaking headlines for a gold/oil trader:
{lines}

For each, give the likely immediate impact on gold and/or oil. Return ONLY JSON:
{{"1": "🟢 Gold / 🔴 Oil: tariffs = risk-off", ...}}
Use 🟢 for up, 🔴 for down, ⚪ for unclear. Max 9 words per item. No invented facts."""
    try:
        text = call_claude(prompt, 500)
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {int(k): str(v) for k, v in data.items()}
    except Exception as e:
        print(f"[warn] flash read failed: {e}", file=sys.stderr)
        return {}


def build_flash(items, reads):
    now = dt.datetime.now(TZ)
    out = [f"⚡ <b>{now:%H:%M}</b>"]
    for i, it in enumerate(items, 1):
        link = html.escape(it["link"], quote=True)
        title = html.escape(it["title"])
        head = f"<b>TRUMP:</b> {title}" if it.get("trump") else title
        line = f"• {_asset_icon(it)} <a href=\"{link}\">{head}</a> <i>· {html.escape(it['source'] or '')} {_ago(it['published'])}</i>"
        if reads.get(i):
            line += f"\n   ↳ {html.escape(reads[i])}"
        out.append(line)
    return "\n".join(out)


def watch(minutes, state_path, poll_s=60, google_every=5, max_age_min=90):
    """Poll fast feeds every minute (Google News every few minutes) and push new items as bullets."""
    end = time.time() + minutes * 60
    state = load_state(state_path)
    prices, prices_at, n = {}, 0, 0
    while time.time() < end:
        n += 1
        try:
            if time.time() - prices_at > 900:                    # refresh prices every 15 min (calendar)
                try:
                    prices, prices_at = fetch_prices(), time.time()
                except Exception as e:
                    print(f"[warn] prices failed: {e}", file=sys.stderr)
                    prices_at = time.time()
            if not os.getenv("NO_CALENDAR") and n % 5 == 1:     # calendar check every ~5 min
                for msg, keys in run_calendar(prices, state):
                    send_telegram(msg)
                    save_state(state_path, state, [], keys)

            items = fetch_news(max_age_min / 60, include_google=(n % google_every == 1))
            fresh = sorted((it for it in items if it["key"] not in state), key=lambda x: x["published"])
            if fresh:
                fresh = fresh[-8:]                               # never flood: newest 8 per poll
                send_telegram(build_flash(fresh, flash_reads(fresh)))
                save_state(state_path, state, fresh)
                print(f"[{dt.datetime.now(TZ):%H:%M:%S}] sent {len(fresh)}")
        except Exception as e:
            print(f"[warn] poll failed: {e}", file=sys.stderr)
        time.sleep(max(0, min(poll_s, end - time.time())))


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print instead of sending")
    ap.add_argument("--lookback", type=int, default=int(os.getenv("LOOKBACK_HOURS", "12")))
    ap.add_argument("--calendar", action="store_true", help="send today's calendar now (testing)")
    ap.add_argument("--no-calendar", action="store_true", help="skip the economic calendar")
    ap.add_argument("--watch", type=float, metavar="MINUTES", default=float(os.getenv("WATCH_MINUTES", "0")),
                    help="keep running for N minutes, pushing new headlines as bullet alerts")
    args = ap.parse_args()

    if args.watch > 0:
        if args.no_calendar:
            os.environ["NO_CALENDAR"] = "1"
        watch(args.watch, os.getenv("STATE_FILE", ".seen.json"))
        return

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
    outgoing = []   # (message, news_items, extra_keys)

    # 1) economic calendar: daily list + heads-up alerts
    if not args.no_calendar:
        for msg, keys in run_calendar(prices, state, force_daily=args.calendar):
            outgoing.append((msg, [], keys))

    # 2) headlines
    news = [it for it in fetch_news(lookback) if it["key"] not in state]
    if news or os.getenv("SKIP_EMPTY") != "1":
        reads, overall = llm_reads(prices, news, lookback)
        outgoing.append((build_message(prices, news, reads, overall, lookback), news, []))

    if not outgoing:
        print("Nothing new - nothing sent.")
        return
    for msg, items, keys in outgoing:
        if args.dry_run:
            print(msg + "\n" + "-" * 40)
        else:
            send_telegram(msg)
            save_state(state_path, state, items, keys)
    print(f"Done: {len(outgoing)} message(s), {len(news)} headlines")


if __name__ == "__main__":
    main()
