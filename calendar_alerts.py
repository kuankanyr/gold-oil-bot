"""
ForexFactory calendar -> Telegram alerts for Richie bot.

Runs once a day (see .github/workflows/calendar-alerts.yml) and sends:
  - "TOMORROW" alert: events happening tomorrow (Bangkok date)  -> 1 day before
  - "TODAY" alert:    events happening today (Bangkok date)     -> event day

Needs two GitHub secrets: TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
"""
import json
import os
import sys
import urllib.request
from datetime import datetime, timedelta, timezone

BKK = timezone(timedelta(hours=7))
FEEDS = [
    "https://nfs.faireconomy.media/ff_calendar_thisweek.json",
    "https://nfs.faireconomy.media/ff_calendar_nextweek.json",  # often 404 until late week; ignored if missing
]

ALL_INSTR = "XAUEUR, Brent, WTI, USOIL, BTCUSD, US500, US30, USTEC"
OIL_INSTR = "Brent, WTI, USOIL"
OIL_WORDS = ("crude", "oil", "opec", "natural gas")

# Which events to alert on
RULES = {
    "USD": {"High"},          # add "Medium" here if you want more alerts
    "EUR": {"High"},
}


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (RichieBot calendar)"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"skip {url}: {e}")
        return []


# ---------------------------------------------------------------------------
# Brief per-asset playbook. First matching rule wins (checked top to bottom).
# "Hot/strong" = above forecast (for Claims: below forecast = strong).
# ---------------------------------------------------------------------------
USD_HAWK = ("⬆️ Hot/strong → USD↑ yields↑ | XAUEUR↓ (less than XAUUSD, weaker EUR cushions) | "
            "USTEC↓↓ US500↓ US30↓ | BTC↓ | Oil slight↓\n"
            "⬇️ Soft/weak → USD↓ | XAUEUR↑ | indices↑ (USTEC most) | BTC↑ | Oil slight↑")
PLAYBOOK = [
    (("crude oil inventories", "eia"), OIL_INSTR,
     "Bigger DRAW than expected → oil↑ | BUILD → oil↓. WTI/USOIL react more than Brent. "
     "Check the API number the night before for a preview. Gold/indices/BTC: little effect."),
    (("opec",), OIL_INSTR,
     "Output cut / quota discipline → oil↑ | output hike or a cheating headline → oil↓. Can gap at the open."),
    (("natural gas",), OIL_INSTR, "Mostly a gas-only mover. Small spillover to oil."),
    (("unemployment claims",), ALL_INSTR,
     "Claims BELOW forecast = strong labor → USD↑ XAUEUR↓ indices↓ BTC↓\n"
     "Claims ABOVE forecast = weak labor → USD↓ XAUEUR↑ indices↑ (rate-cut hopes)"),
    (("non-farm", "nonfarm", "employment change", "unemployment rate"), ALL_INSTR,
     USD_HAWK + "\n(Unemployment rate is inverted: higher = weak.) First spike often reverses. Wait for the 15m close."),
    (("cpi", "pce", "ppi"), ALL_INSTR,
     USD_HAWK + "\nWatch the CORE m/m number most. Wide spreads at release."),
    (("retail sales",), ALL_INSTR,
     USD_HAWK + "\nStrong consumer demand also supports oil slightly."),
    (("gdp", "ism", "pmi", "philly", "empire"), ALL_INSTR, USD_HAWK),
    (("fomc", "federal funds", "fed chair", "powell", "warsh"), ALL_INSTR,
     "Hawkish (cuts later / inflation worry) → USD↑ | XAUEUR↓ | USTEC↓↓ US500↓ US30↓ | BTC↓ | oil slight↓\n"
     "Dovish (cuts sooner) → reverse. Moves come headline by headline. Avoid tight stops at release."),
    (("lagarde", "ecb", "main refinancing", "deposit facility"), "XAUEUR",
     "Hawkish → EUR↑ → XAUEUR↓ | Dovish → EUR↓ → XAUEUR↑. USD assets: little effect."),
]
EUR_DEFAULT = ("XAUEUR", "Strong EUR data → EUR↑ → XAUEUR↓ | Weak → EUR↓ → XAUEUR↑. USD assets: little effect.")
USD_DEFAULT = (ALL_INSTR, USD_HAWK)


def playbook_for(ev):
    title = ev.get("title", "").lower()
    for words, instr, note in PLAYBOOK:
        if any(w in title for w in words):
            # EUR versions of CPI/PMI/GDP etc. → EUR logic
            if ev.get("country") == "EUR" and instr == ALL_INSTR:
                return EUR_DEFAULT
            return instr, note
    return EUR_DEFAULT if ev.get("country") == "EUR" else USD_DEFAULT


def wanted(ev):
    title = ev.get("title", "").lower()
    if ev.get("country") in ("USD", "ALL") and any(w in title for w in OIL_WORDS):
        return True
    return ev.get("impact") in RULES.get(ev.get("country"), set())


def load_events():
    seen, out = set(), []
    for url in FEEDS:
        for ev in fetch(url):
            key = (ev.get("title"), ev.get("country"), ev.get("date"))
            if key in seen or not wanted(ev):
                continue
            seen.add(key)
            try:
                ev["_dt"] = datetime.fromisoformat(ev["date"]).astimezone(BKK)
            except Exception:  # noqa: BLE001
                continue
            out.append(ev)
    return sorted(out, key=lambda e: e["_dt"])


def numbers(ev):
    f, p = ev.get("forecast") or "", ev.get("previous") or ""
    return f"F: {f or '–'} | P: {p or '–'}" if (f or p) else ""


def fmt_group(g):
    """One block per time slot: headline event, extra releases, then the playbook once."""
    ev = g[0]
    icon = "🔴" if any(e.get("impact") == "High" for e in g) else "🟠"
    instr, note = playbook_for(ev)
    lines = [f"{icon} <b>{ev['_dt'].strftime('%H:%M')}</b> {ev['country']} – {ev['title']}"]
    if numbers(ev):
        lines.append(f"      {numbers(ev)}")
    for e in g[1:]:
        lines.append(f"   + {e['title']}" + (f"  ({numbers(e)})" if numbers(e) else ""))
    lines.append(f"      ↳ {instr}")
    lines += [f"      {l}" for l in note.split("\n")]
    return "\n".join(lines)


def group(evs):
    """Bundle releases with the same time + currency + playbook into one block."""
    out = []
    for e in evs:
        key = (e["_dt"], e["country"], playbook_for(e)[1])
        if out and out[-1][0] == key:
            out[-1][1].append(e)
        else:
            out.append((key, [e]))
    return [g for _, g in out]


def send(text):
    token = os.environ["TELEGRAM_BOT_TOKEN"]
    chat = os.environ["TELEGRAM_CHAT_ID"]
    data = json.dumps({"chat_id": chat, "text": text, "parse_mode": "HTML",
                       "disable_web_page_preview": True}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage",
                                 data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        print("telegram:", r.status)


def main():
    events = load_events()
    today = datetime.now(BKK).date()
    tomorrow = today + timedelta(days=1)

    for label, day in (("📅 TODAY", today), ("⏰ TOMORROW", tomorrow)):
        todays = [e for e in events if e["_dt"].date() == day]
        if not todays:
            print(f"{label}: nothing")
            continue
        head = f"<b>{label} – {day.strftime('%a %d %b')}</b> (Bangkok time)\n"
        body = "\n\n".join(fmt_group(g) for g in group(todays))
        if "--dry-run" in sys.argv:
            print(head + body)
        else:
            msg = head + "\n" + body
            while msg:
                cut = msg[:4000].rfind("\n\n") if len(msg) > 4000 else len(msg)
                cut = cut if cut > 0 else 4000
                send(msg[:cut])
                msg = msg[cut:].lstrip()


if __name__ == "__main__":
    main()
