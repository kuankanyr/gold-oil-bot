"""
ForexFactory -> Google Calendar sync (runs in GitHub Actions after the Telegram alerts).

- Reads the same events + per-asset playbook as calendar_alerts.py
- One calendar event per time slot per currency (e.g. all 19:30 USD releases together)
- Safe to run every day: events are created once, then updated in place
  (forecasts refresh; no duplicates). Slots that already have a 🔴/🟠 event
  you or Claude added by hand are left alone.
- Reminders: popup 1 day before + 1 hour before.

Needs GitHub secrets:
  GOOGLE_SERVICE_ACCOUNT_JSON  – full JSON key of the service account
  GOOGLE_CALENDAR_ID           – your calendar ID (your Gmail address)
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timedelta

import calendar_alerts as ca

API = "https://www.googleapis.com/calendar/v3/calendars/{cal}/events"
SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def session():
    from google.oauth2 import service_account
    from google.auth.transport.requests import AuthorizedSession
    info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
    creds = service_account.Credentials.from_service_account_info(info, scopes=SCOPES)
    return AuthorizedSession(creds)


def slots(events):
    """Group by (Bangkok time, currency) – merge all releases in that slot."""
    out = {}
    for e in events:
        key = (e["_dt"], "OIL" if any(w in e["title"].lower() for w in ca.OIL_WORDS) else e["country"])
        out.setdefault(key, []).append(e)
    return out


def build(key, evs):
    dt, ccy = key
    high = any(e.get("impact") == "High" for e in evs)
    icon = "🔴" if high else "🟠"
    names = []
    for e in evs:
        n = e["title"]
        if n not in names:
            names.append(n)
    title = ", ".join(names)
    if len(title) > 60:
        title = title[:57] + "…"
    instr_set, notes = [], []
    for e in evs:
        instr, note = ca.playbook_for(e)
        if instr not in instr_set:
            instr_set.append(instr)
        if note not in notes:
            notes.append(note)
    lines = [f"Impact: {'High' if high else 'Medium'} ({evs[0]['country']})",
             f"Affects: {max(instr_set, key=len)}", ""]
    for e in evs:
        n = ca.numbers(e)
        lines.append(f"{e['title']}" + (f": {n.replace('F:', 'forecast').replace('P:', 'prev')}" if n else ""))
    seen = []
    for n in notes:
        for l in n.split("\n"):
            if l not in seen:
                seen.append(l)
    lines += ["", "HOW IT MOVES YOUR ASSETS"] + seen
    if len(evs) >= 3 and high:
        lines += ["", "TRADE NOTE: several releases at once – expect whipsaw; consider being flat."]
    lines += ["", "Source: ForexFactory calendar · auto-synced by Richie bot"]
    end = dt + timedelta(minutes=30)
    return {
        "id": "ff" + hashlib.md5(f"{ccy}|{dt.isoformat()}".encode()).hexdigest(),
        "summary": f"{icon} {ccy} – {title}",
        "description": "\n".join(lines),
        "start": {"dateTime": dt.isoformat(), "timeZone": "Asia/Bangkok"},
        "end": {"dateTime": end.isoformat(), "timeZone": "Asia/Bangkok"},
        "colorId": "11" if high else "6",
        "transparency": "transparent",
        "reminders": {"useDefault": False, "overrides": [
            {"method": "popup", "minutes": 1440}, {"method": "popup", "minutes": 60}]},
    }


def existing_manual(s, url, tmin, tmax):
    """Start times of 🔴/🟠 events NOT created by this script (hand-added ones)."""
    taken, page = set(), None
    while True:
        params = {"timeMin": tmin, "timeMax": tmax, "singleEvents": "true", "maxResults": 250}
        if page:
            params["pageToken"] = page
        r = s.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        for ev in data.get("items", []):
            if ev.get("id", "").startswith("ff"):
                continue
            if ev.get("summary", "")[:1] in ("🔴", "🟠") and "dateTime" in ev.get("start", {}):
                taken.add(datetime.fromisoformat(ev["start"]["dateTime"]).astimezone(ca.BKK))
        page = data.get("nextPageToken")
        if not page:
            return taken


def main():
    events = [e for e in ca.load_events() if e["_dt"] >= datetime.now(ca.BKK) - timedelta(hours=1)]
    bodies = [build(k, v) for k, v in sorted(slots(events).items())]
    if "--dry-run" in sys.argv:
        for b in bodies:
            print(b["start"]["dateTime"], b["summary"], "\n", b["description"], "\n")
        return
    if not bodies:
        print("no upcoming events")
        return
    s = session()
    url = API.format(cal=os.environ["GOOGLE_CALENDAR_ID"])
    tmin = min(b["start"]["dateTime"] for b in bodies)
    tmax = max(b["end"]["dateTime"] for b in bodies)
    manual = existing_manual(s, url, tmin, tmax)
    made = updated = skipped = 0
    for b in bodies:
        if datetime.fromisoformat(b["start"]["dateTime"]) in manual:
            skipped += 1
            continue
        r = s.put(f"{url}/{b['id']}", json=b)          # update if it exists
        if r.status_code == 404:
            r = s.post(url, json=b)                     # otherwise create
            made += 1
        else:
            updated += 1
        if r.status_code >= 400:
            print("error", r.status_code, r.text[:300])
            r.raise_for_status()
    print(f"calendar: {made} created, {updated} updated, {skipped} skipped (already added by hand)")


if __name__ == "__main__":
    main()
