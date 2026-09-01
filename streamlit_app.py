from textwrap import dedent
from pathlib import Path
from agno.agent import Agent
import streamlit as st
import re
from types import SimpleNamespace
from agno.models.openai import OpenAIChat
from agno.models.openrouter import OpenRouter
from agno.models.groq import Groq
from agno.models.xai import xAI
from groq import Groq as GroqSDK
from icalendar import Calendar, Event
from datetime import datetime, timedelta
import requests
import json
import time
import os
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

# #region debug logging
LOG_PATH = "/Users/srilaxmich/Desktop/Generative-AI-Projects/.cursor/debug.log"
def debug_log(location, message, data=None, hypothesis_id=None, session_id="debug-session", run_id="run1"):
    try:
        log_entry = {
            "id": f"log_{int(time.time() * 1000)}",
            "timestamp": int(time.time() * 1000),
            "location": location,
            "message": message,
            "data": data or {},
            "sessionId": session_id,
            "runId": run_id,
            "hypothesisId": hypothesis_id
        }
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception:
        pass
# #endregion

def duckduckgo_search(query: str, max_results: int = 6, per_site_timeout: float = 8.0, enrich: bool = True):
    """Lightweight DuckDuckGo search with optional snippet enrichment."""
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title"),
                    "url": r.get("href"),
                    "snippet": r.get("body"),
                })
    except Exception as e:
        debug_log("travel_agent.py:duckduckgo_search", "ddg error", {"error": str(e)}, "A")
        return [{"title": "Search error", "url": None, "snippet": str(e)}]

    if not enrich:
        return results

    enriched = []
    headers = {"User-Agent": "Mozilla/5.0"}
    for r in results[:3]:
        url = r.get("url")
        snippet = r.get("snippet") or ""
        if not url:
            enriched.append(r)
            continue
        try:
            resp = requests.get(url, timeout=per_site_timeout, headers=headers)
            if resp.status_code == 200:
                soup = BeautifulSoup(resp.text, "html.parser")
                text = " ".join(soup.get_text(" ", strip=True).split()[:120])
                snippet = snippet or text[:400]
        except Exception:
            pass
        enriched.append({**r, "snippet": snippet})
    enriched += results[3:]
    return enriched


def build_research_summary(destination: str, num_days: int, travel_style: str, budget_range: str, interests, enrich: bool = True, max_results: int = 6):
    """Generate a concise research summary using DuckDuckGo (no API key)."""
    queries = [
        f"{destination} {travel_style} travel {num_days} days {budget_range}",
        f"{destination} top things to do {budget_range}",
    ]
    if interests:
        queries.append(f"{destination} {', '.join(interests)} best spots")

    summaries = []
    for q in queries[:2]:
        hits = duckduckgo_search(q, max_results=max_results, enrich=enrich)
        for h in hits[:4]:
            title = h.get("title") or "Result"
            url = h.get("url") or ""
            snippet = h.get("snippet") or ""
            summaries.append(f"- {title} — {snippet} ({url})")
    return "\n".join(summaries) if summaries else "No results found."


def parse_cities(raw: str):
    """Split city input on new lines, arrows, or lists. Keep 'City, Country' as one place."""
    if not raw:
        return []
    chunks = re.split(r"[\n;|]+|->|→", raw)
    chunks = [" ".join(part.strip().split()) for part in chunks if part.strip()]
    if len(chunks) == 1 and "," in chunks[0]:
        parts = [p.strip() for p in chunks[0].split(",") if p.strip()]
        if len(parts) == 2:
            chunks = [chunks[0]]
        elif len(parts) >= 4 and len(parts) % 2 == 0:
            chunks = [f"{parts[i]}, {parts[i + 1]}" for i in range(0, len(parts), 2)]
        else:
            chunks = parts
    cities = []
    seen = set()
    for city in chunks:
        key = city.lower()
        if not city or key in seen:
            continue
        seen.add(key)
        cities.append(city)
        if len(cities) >= 4:
            break
    return cities


def allocate_city_days(cities, num_days: int):
    """Spread trip days across cities in visit order."""
    if not cities:
        return []
    usable_cities = cities[: max(1, min(len(cities), num_days))]
    n = len(usable_cities)
    base = num_days // n
    extra = num_days % n
    plan = []
    day = 1
    for i, city in enumerate(usable_cities):
        days = base + (1 if i < extra else 0)
        start_day = day
        end_day = day + days - 1
        plan.append({
            "city": city,
            "days": days,
            "start_day": start_day,
            "end_day": end_day,
        })
        day = end_day + 1
    return plan


def format_city_plan(plan) -> str:
    if not plan:
        return ""
    return "\n".join(
        f"- {p['city']}: Days {p['start_day']}-{p['end_day']} ({p['days']} days)"
        for p in plan
    )


def destination_label(plan) -> str:
    if not plan:
        return "your trip"
    return " → ".join(p["city"] for p in plan)


def build_multi_city_research(plan, travel_style: str, budget_range: str, interests):
    chunks = []
    for p in plan:
        summary = build_research_summary(
            p["city"], p["days"], travel_style, budget_range, interests,
            enrich=False, max_results=4,
        )
        chunks.append(f"{p['city']}:\n{truncate_content(summary, 450)}")
    return "\n\n".join(chunks)


def normalize_hotel_key(text: str) -> str:
    """Collapse listing titles so the same property is not listed twice."""
    t = (text or "").lower()
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"^[\-\*\d\.\)\s]+", "", t)
    t = re.sub(r"^[^:]{2,40}:\s*", "", t, count=1)
    t = re.sub(
        r"\b(booking\.com|tripadvisor|hotels\.com|makemytrip|goibibo|agoda|expedia|trivago|kayak)\b",
        " ",
        t,
    )
    t = re.sub(r"\b(reviews?|prices?|deals?|photos?|official site|from \d+)\b", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())[:80]


def collect_hotel_hits(plan, budget_range: str, nightly_min=None, nightly_max=None, currency="INR"):
    """Search real hotel listings per city for prompts and fallbacks."""
    out = []
    ccy = currency or "INR"
    lo = nightly_min
    hi = nightly_max
    for p in plan:
        queries = [
            f"{p['city']} hotels {budget_range} {lo}-{hi} {ccy} per night",
            f"{p['city']} guesthouse hotel under {hi} {ccy}",
        ]
        if budget_range == "Budget-Friendly":
            queries.extend([
                f"{p['city']} budget hotel hostel homestay cheap stay",
                f"{p['city']} 2 star 3 star hotel downtown affordable",
            ])
        elif budget_range == "Mid-Range":
            queries.append(f"{p['city']} 3 star 4 star hotel {lo}-{hi} {ccy}")
        else:
            queries.append(f"{p['city']} luxury boutique hotel {lo}-{hi} {ccy}")

        seen = set()
        cleaned = []
        for query in queries:
            hits = duckduckgo_search(query, max_results=5, enrich=False)
            for h in hits or []:
                title = (h.get("title") or "").strip()
                key = normalize_hotel_key(title)
                if not title or title.lower() == "search error" or key in seen:
                    continue
                seen.add(key)
                cleaned.append({
                    "title": title,
                    "url": h.get("url") or "",
                    "snippet": (h.get("snippet") or "")[:180],
                })
            if len(cleaned) >= 8:
                break
        out.append({"city": p["city"], "hits": cleaned[:8]})
    return out


def format_hotel_research_notes(hotel_hits) -> str:
    chunks = []
    for item in hotel_hits or []:
        lines = []
        for h in item.get("hits") or []:
            url = f" ({h['url']})" if h.get("url") else ""
            snippet = f" — {h['snippet']}" if h.get("snippet") else ""
            lines.append(f"- {h['title']}{snippet}{url}")
        chunks.append(f"{item['city']}:\n" + ("\n".join(lines) if lines else "- No hotel listings found"))
    return "\n\n".join(chunks)


def fallback_hotels_markdown(hotel_hits, currency: str, nightly_min=None, nightly_max=None) -> str:
    """Deterministic hotel list from search hits when the model leaks reasoning."""
    rec_lines = []
    alt_lines = []
    ccy = currency or "INR"
    range_note = ""
    if nightly_min is not None and nightly_max is not None:
        range_note = f" Aim for about {ccy} {int(nightly_min)}–{int(nightly_max)} per night."
    used = set()
    for item in hotel_hits or []:
        city = item.get("city") or "City"
        unique_hits = []
        for hit in item.get("hits") or []:
            key = normalize_hotel_key(hit.get("title") or "")
            if not key or key in used:
                continue
            used.add(key)
            unique_hits.append(hit)
        recs = unique_hits[:3]
        alts = unique_hits[3:6]
        if not recs:
            rec_lines.append(_format_hotel_bullet(city, None, ccy, "central stay") + range_note)
            continue
        for hit in recs:
            rec_lines.append(_format_hotel_bullet(city, hit, ccy, "in-budget stay") + range_note)
        for hit in alts:
            alt_lines.append(_format_hotel_bullet(city, hit, ccy, "backup stay") + range_note)
    recommended = "Recommended\n" + "\n".join(rec_lines)
    alternative = ("Alternative\n" + "\n".join(alt_lines)) if alt_lines else ""
    return f"{recommended}\n\n{alternative}".strip()


def _format_hotel_bullet(city, hit, currency, kind):
    if not hit:
        return (
            f"- {city}: Search Booking.com for a well-reviewed {kind} near the center. "
            f"Confirm the nightly rate in {currency} before booking. https://www.booking.com"
        )
    snippet = f" {hit['snippet']}" if hit.get("snippet") else ""
    url = hit.get("url") or "https://www.booking.com"
    return (
        f"- {city}: {hit['title']}.{snippet} "
        f"Typical nightly rate in {currency} (confirm on the booking site). {url}"
    )


def build_hotel_research(plan, budget_range: str):
    return format_hotel_research_notes(collect_hotel_hits(plan, budget_range))


def estimate_trip_budget(budget_range: str, num_days: int, city_count: int, travelers: int, lodging_night_usd=None) -> dict:
    """Deterministic USD ranges so budgets can be converted to any currency."""
    rates = {
        "Budget-Friendly": {
            "lodging": (45, 85), "food": (20, 40), "activities": (15, 35),
            "local": (8, 18), "intercity": (30, 80),
        },
        "Mid-Range": {
            "lodging": (95, 190), "food": (40, 85), "activities": (35, 75),
            "local": (12, 28), "intercity": (45, 130),
        },
        "Luxury": {
            "lodging": (230, 520), "food": (90, 190), "activities": (80, 180),
            "local": (25, 65), "intercity": (90, 260),
        },
    }
    band = dict(rates.get(budget_range, rates["Mid-Range"]))
    if lodging_night_usd and lodging_night_usd[0] > 0 and lodging_night_usd[1] > 0:
        band["lodging"] = (min(lodging_night_usd), max(lodging_night_usd))
    nights = max(num_days - 1, 1)
    hops = max(city_count - 1, 0)
    people = max(travelers, 1)
    rooms = max(1, (people + 1) // 2)

    def span(lo, hi, qty, count=people):
        return lo * qty * count, hi * qty * count

    lodging = span(*band["lodging"], nights, rooms)
    food = span(*band["food"], num_days)
    activities = span(*band["activities"], num_days)
    local = span(*band["local"], num_days)
    intercity = span(*band["intercity"], hops) if hops else (0, 0)
    total = (
        lodging[0] + food[0] + activities[0] + local[0] + intercity[0],
        lodging[1] + food[1] + activities[1] + local[1] + intercity[1],
    )
    return {
        "people": people,
        "budget_range": budget_range,
        "num_days": num_days,
        "nights": nights,
        "city_count": city_count,
        "hops": hops,
        "lodging_night": band["lodging"],
        "lodging": lodging,
        "food": food,
        "activities": activities,
        "local": local,
        "intercity": intercity,
        "total": total,
        "per_person": (total[0] / people, total[1] / people),
    }


CURRENCY_SYMBOLS = {
    "USD": "$", "EUR": "€", "GBP": "£", "INR": "₹", "JPY": "¥", "AUD": "A$",
    "CAD": "C$", "CHF": "CHF ", "AED": "AED ", "SGD": "S$", "THB": "฿",
    "MXN": "MX$", "BRL": "R$", "KRW": "₩", "CNY": "¥", "ZAR": "R",
    "NZD": "NZ$", "HKD": "HK$", "IDR": "Rp", "PHP": "₱", "TRY": "₺",
    "PLN": "zł", "SEK": "kr", "NOK": "kr", "DKK": "kr", "SAR": "SAR ",
    "QAR": "QAR ", "EGP": "E£", "VND": "₫", "NPR": "Rs", "LKR": "Rs",
    "MYR": "RM", "ILS": "₪", "CZK": "Kč", "PKR": "Rs", "BDT": "৳",
}

CURRENCY_OPTIONS = [
    "INR", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "AED", "SGD",
    "THB", "MXN", "BRL", "KRW", "CNY", "ZAR", "NZD", "HKD", "IDR", "PHP",
    "TRY", "PLN", "SEK", "NOK", "DKK", "SAR", "QAR", "EGP", "VND", "NPR",
    "LKR", "MYR", "ILS", "CZK", "PKR", "BDT",
]

ISO_TO_CURRENCY = {
    "US": "USD", "GB": "GBP", "IN": "INR", "JP": "JPY", "FR": "EUR", "DE": "EUR",
    "IT": "EUR", "ES": "EUR", "PT": "EUR", "NL": "EUR", "BE": "EUR", "IE": "EUR",
    "AT": "EUR", "GR": "EUR", "FI": "EUR", "AU": "AUD", "CA": "CAD", "CH": "CHF",
    "AE": "AED", "SG": "SGD", "TH": "THB", "MX": "MXN", "BR": "BRL", "KR": "KRW",
    "CN": "CNY", "ZA": "ZAR", "NZ": "NZD", "HK": "HKD", "ID": "IDR", "PH": "PHP",
    "TR": "TRY", "PL": "PLN", "SE": "SEK", "NO": "NOK", "DK": "DKK", "SA": "SAR",
    "QA": "QAR", "EG": "EGP", "VN": "VND", "NP": "NPR", "LK": "LKR", "BD": "BDT",
    "PK": "PKR", "MY": "MYR", "KE": "KES", "NG": "NGN", "AR": "ARS", "CL": "CLP",
    "PE": "PEN", "CO": "COP", "CZ": "CZK", "HU": "HUF", "RO": "RON", "UA": "UAH",
    "IL": "ILS", "MA": "MAD", "TN": "TND", "KW": "KWD", "BH": "BHD", "OM": "OMR",
}

PLACE_CURRENCY_HINTS = {
    "paris": "EUR", "lyon": "EUR", "nice": "EUR", "france": "EUR",
    "rome": "EUR", "milan": "EUR", "florence": "EUR", "venice": "EUR", "italy": "EUR",
    "barcelona": "EUR", "madrid": "EUR", "spain": "EUR", "berlin": "EUR", "germany": "EUR",
    "amsterdam": "EUR", "lisbon": "EUR", "athens": "EUR", "vienna": "EUR", "dublin": "EUR",
    "tokyo": "JPY", "osaka": "JPY", "kyoto": "JPY", "japan": "JPY",
    "london": "GBP", "edinburgh": "GBP", "manchester": "GBP", "uk": "GBP", "england": "GBP",
    "scotland": "GBP", "united kingdom": "GBP",
    "mumbai": "INR", "delhi": "INR", "bengaluru": "INR", "bangalore": "INR", "goa": "INR",
    "jaipur": "INR", "kerala": "INR", "india": "INR",
    "new york": "USD", "los angeles": "USD", "chicago": "USD", "miami": "USD",
    "san francisco": "USD", "usa": "USD", "united states": "USD",
    "dubai": "AED", "abu dhabi": "AED", "uae": "AED",
    "bangkok": "THB", "phuket": "THB", "thailand": "THB",
    "sydney": "AUD", "melbourne": "AUD", "australia": "AUD",
    "toronto": "CAD", "vancouver": "CAD", "canada": "CAD",
    "singapore": "SGD", "seoul": "KRW", "korea": "KRW", "south korea": "KRW",
    "mexico city": "MXN", "mexico": "MXN", "rio": "BRL", "brazil": "BRL",
    "bali": "IDR", "jakarta": "IDR", "indonesia": "IDR",
    "istanbul": "TRY", "turkey": "TRY", "prague": "CZK", "czech": "CZK",
    "cairo": "EGP", "egypt": "EGP", "hanoi": "VND", "vietnam": "VND",
    "kathmandu": "NPR", "nepal": "NPR", "colombo": "LKR", "sri lanka": "LKR",
    "kuala lumpur": "MYR", "malaysia": "MYR", "tel aviv": "ILS", "israel": "ILS",
}


def geocode_country_code(place: str):
    try:
        query = place.split(",")[-1].strip() or place
        response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": query, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        if response.status_code != 200:
            return None
        results = (response.json() or {}).get("results") or []
        if not results:
            return None
        return (results[0].get("country_code") or "").upper() or None
    except Exception:
        return None


def infer_currency_from_places(places) -> str:
    """Pick a local currency from city/country/state names."""
    for place in places or []:
        blob = f" {place.lower()} "
        for hint, code in PLACE_CURRENCY_HINTS.items():
            if f" {hint} " in blob or place.lower().endswith(hint) or place.lower().startswith(hint):
                return code
        country = geocode_country_code(place)
        if country and country in ISO_TO_CURRENCY:
            return ISO_TO_CURRENCY[country]
    return "INR"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_usd_rates():
    rates = {"USD": 1.0}
    try:
        response = requests.get("https://open.er-api.com/v6/latest/USD", timeout=8)
        if response.status_code == 200:
            payload = response.json() or {}
            if payload.get("result") == "success":
                rates.update(payload.get("rates") or {})
    except Exception:
        pass
    return rates


def usd_to_currency(amount_usd: float, currency: str, rates=None) -> float:
    rates = rates or fetch_usd_rates()
    rate = float(rates.get(currency, 1.0) or 1.0)
    return amount_usd * rate


DEFAULT_NIGHTLY_USD = {
    "Budget-Friendly": (30, 70),
    "Mid-Range": (80, 170),
    "Luxury": (200, 480),
}


def currency_to_usd(amount, currency, rates=None) -> float:
    rates = rates or fetch_usd_rates()
    rate = float(rates.get(currency, 1.0) or 1.0)
    if rate <= 0:
        return float(amount)
    return float(amount) / rate


def convert_amount(amount, from_ccy, to_ccy, rates=None) -> float:
    if not from_ccy or from_ccy == to_ccy:
        return float(amount)
    rates = rates or fetch_usd_rates()
    return usd_to_currency(currency_to_usd(amount, from_ccy, rates), to_ccy, rates)


def _round_local_money(amount, currency) -> int:
    value = max(1, float(amount))
    if currency in {"JPY", "KRW", "VND", "IDR"}:
        return max(1, int(round(value / 100.0) * 100))
    if currency in {"INR", "THB", "PHP", "PKR", "NPR"}:
        return max(1, int(round(value / 50.0) * 50))
    return max(1, int(round(value)))


def nightly_defaults_local(band, currency):
    lo, hi = DEFAULT_NIGHTLY_USD.get(band, DEFAULT_NIGHTLY_USD["Mid-Range"])
    rates = fetch_usd_rates()
    return (
        _round_local_money(usd_to_currency(lo, currency, rates), currency),
        _round_local_money(usd_to_currency(hi, currency, rates), currency),
    )


def money(amount_usd: float, currency: str, rates=None) -> str:
    converted = usd_to_currency(amount_usd, currency, rates)
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    if currency in {"JPY", "KRW", "VND", "IDR"}:
        return f"{symbol}{converted:,.0f}"
    return f"{symbol}{converted:,.0f}"


def format_local_amount(amount: float, currency: str) -> str:
    symbol = CURRENCY_SYMBOLS.get(currency, f"{currency} ")
    if currency in {"JPY", "KRW", "VND", "IDR"} or amount >= 20:
        return f"{symbol}{amount:,.0f}"
    return f"{symbol}{amount:,.2f}"


def convert_prose_currency(text, from_ccy, to_ccy):
    """Convert amounts that already include a currency code or symbol."""
    if not text or not from_ccy or not to_ccy or from_ccy == to_ccy:
        return text
    rates = fetch_usd_rates()
    src = float(rates.get(from_ccy) or 0)
    dst = float(rates.get(to_ccy) or 0)
    if src <= 0 or dst <= 0:
        return text
    factor = dst / src
    symbol = (CURRENCY_SYMBOLS.get(from_ccy) or "").strip()
    number = r"[0-9][0-9,]*(?:\.[0-9]+)?"

    def as_local(raw):
        value = float(str(raw).replace(",", "")) * factor
        return format_local_amount(value, to_ccy)

    def range_repl(match):
        try:
            return f"{as_local(match.group(1))}–{as_local(match.group(2))}"
        except ValueError:
            return match.group(0)

    def one_repl(match):
        try:
            return as_local(match.group(1))
        except ValueError:
            return match.group(0)

    out = text
    out = re.sub(
        rf"(?i)\b{re.escape(from_ccy)}\s*({number})\s*[-–—]\s*(?:{re.escape(from_ccy)}\s*)?({number})",
        range_repl,
        out,
    )
    out = re.sub(
        rf"(?i)({number})\s*[-–—]\s*({number})\s*{re.escape(from_ccy)}\b",
        range_repl,
        out,
    )
    out = re.sub(rf"(?i)\b{re.escape(from_ccy)}\s*({number})", one_repl, out)
    out = re.sub(rf"(?i)({number})\s*{re.escape(from_ccy)}\b", one_repl, out)
    if symbol and (len(symbol) > 1 or symbol in {"€", "£", "¥", "₹", "₩", "฿", "₱", "₺", "₫", "$"}):
        out = re.sub(
            rf"{re.escape(symbol)}\s*({number})\s*[-–—]\s*(?:{re.escape(symbol)}\s*)?({number})",
            range_repl,
            out,
        )
        out = re.sub(rf"{re.escape(symbol)}\s*({number})", one_repl, out)
    if from_ccy == "USD":
        out = re.sub(rf"\$\s*({number})\s*[-–—]\s*\$?\s*({number})", range_repl, out)
        out = re.sub(rf"\$\s*({number})", one_repl, out)
    return out


def text_in_display_currency(text):
    return convert_prose_currency(
        text or "",
        st.session_state.get("price_currency") or st.session_state.get("detected_currency") or "INR",
        st.session_state.get("display_currency") or "INR",
    )


def format_budget_estimate(data: dict, currency: str) -> str:
    if not data:
        return ""
    rates = fetch_usd_rates()
    ccy = currency or "INR"
    return dedent(f"""\
        Baseline estimate in {ccy} for {data['people']} traveler(s), {data['budget_range']}, {data['num_days']} days / {data['nights']} nights, {data['city_count']} city(ies).
        - Lodging: {money(data['lodging'][0], ccy, rates)}–{money(data['lodging'][1], ccy, rates)} ({money(data['lodging_night'][0], ccy, rates)}–{money(data['lodging_night'][1], ccy, rates)}/room-night × {data['nights']} nights)
        - Food: {money(data['food'][0], ccy, rates)}–{money(data['food'][1], ccy, rates)}
        - Activities: {money(data['activities'][0], ccy, rates)}–{money(data['activities'][1], ccy, rates)}
        - Local transport: {money(data['local'][0], ccy, rates)}–{money(data['local'][1], ccy, rates)}
        - Intercity travel ({data['hops']} hop(s)): {money(data['intercity'][0], ccy, rates)}–{money(data['intercity'][1], ccy, rates)}
        - Estimated total: {money(data['total'][0], ccy, rates)}–{money(data['total'][1], ccy, rates)}
        - Per person: {money(data['per_person'][0], ccy, rates)}–{money(data['per_person'][1], ccy, rates)}
        These are planning ranges, not live quotes. Converted from a USD baseline with current FX rates.
    """)


def mark_currency_manual():
    st.session_state.currency_manual = True

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException("Operation timed out")

def run_agent_with_timeout(agent, prompt, timeout_seconds=120, agent_name="agent"):
    """
    Run an agent with timeout protection to prevent hanging.
    Uses ThreadPoolExecutor with timeout, and falls back to direct call with shorter timeout.
    
    Args:
        agent: The agent instance to run
        prompt: The prompt to pass to the agent
        timeout_seconds: Maximum time to wait (default 2 minutes for faster feedback)
        agent_name: Name of agent for error messages
    
    Returns:
        The agent run result
    
    Raises:
        TimeoutError: If the agent call exceeds timeout_seconds
        Exception: Any other exception from the agent
    """
    # #region agent log
    debug_log("travel_agent.py:run_agent_with_timeout", f"Starting {agent_name} with timeout", {"timeout_seconds": timeout_seconds}, "A")
    start_time = time.time()
    # #endregion
    
    def _run():
        # #region agent log
        debug_log("travel_agent.py:_run", f"Inside _run for {agent_name}", {}, "A")
        run_start = time.time()
        # #endregion
        try:
            result = _run_agent_prompt(agent, prompt)
            # #region agent log
            run_elapsed = time.time() - run_start
            debug_log("travel_agent.py:_run", f"{agent_name} _run completed", {"elapsed_seconds": run_elapsed}, "A")
            # #endregion
            return result
        except Exception as e:
            # #region agent log
            run_elapsed = time.time() - run_start
            debug_log("travel_agent.py:_run", f"{agent_name} _run exception", {"error": str(e), "error_type": type(e).__name__, "elapsed_seconds": run_elapsed}, "A")
            # #endregion
            raise
    
    try:
        # #region agent log
        debug_log("travel_agent.py:run_agent_with_timeout", f"Creating ThreadPoolExecutor for {agent_name}", {}, "A")
        # #endregion
        with ThreadPoolExecutor(max_workers=1) as executor:
            # #region agent log
            debug_log("travel_agent.py:run_agent_with_timeout", f"Submitting {agent_name} task to executor", {}, "A")
            submit_start = time.time()
            # #endregion
            future = executor.submit(_run)
            # #region agent log
            submit_elapsed = time.time() - submit_start
            debug_log("travel_agent.py:run_agent_with_timeout", f"Task submitted, waiting for {agent_name} result", {"submit_elapsed": submit_elapsed, "timeout_seconds": timeout_seconds}, "A")
            # #endregion

            last_heartbeat = time.time()
            while True:
                if future.done():
                    result = future.result()
                    # #region agent log
                    total_elapsed = time.time() - start_time
                    debug_log("travel_agent.py:run_agent_with_timeout", f"{agent_name} completed successfully", {"total_elapsed_seconds": total_elapsed}, "A")
                    # #endregion
                    return result

                elapsed = time.time() - start_time
                if elapsed >= timeout_seconds:
                    # #region agent log
                    debug_log("travel_agent.py:run_agent_with_timeout", f"{agent_name} timeout reached", {"timeout_seconds": timeout_seconds, "actual_elapsed": elapsed}, "A")
                    # #endregion
                    raise TimeoutError(f"{agent_name} execution exceeded {timeout_seconds} seconds. This usually means:\n- API keys may be invalid or expired\n- Network connection issues\n- API service is slow or unavailable\n\nPlease check your API keys and try again.")

                if time.time() - last_heartbeat >= 15:
                    # #region agent log
                    debug_log("travel_agent.py:run_agent_with_timeout", f"{agent_name} still running", {"elapsed_seconds": elapsed}, "A")
                    # #endregion
                    last_heartbeat = time.time()
                time.sleep(2)
    except FutureTimeoutError as e:
        # #region agent log
        total_elapsed = time.time() - start_time
        debug_log("travel_agent.py:run_agent_with_timeout", f"{agent_name} ThreadPoolExecutor timeout", {"timeout_seconds": timeout_seconds, "actual_elapsed": total_elapsed, "error": str(e)}, "A")
        # #endregion
        raise TimeoutError(f"{agent_name} execution exceeded {timeout_seconds} seconds. This usually means:\n- API keys may be invalid or expired\n- Network connection issues\n- API service is slow or unavailable\n\nPlease check your API keys and try again.")
    except Exception as e:
        # #region agent log
        total_elapsed = time.time() - start_time
        debug_log("travel_agent.py:run_agent_with_timeout", f"{agent_name} error", {"error": str(e), "error_type": type(e).__name__, "elapsed_seconds": total_elapsed}, "A")
        # #endregion
        raise

def _run_agent_prompt(agent, prompt: str):
    """Run an agent, using the Groq SDK directly for Groq models.

    Agno + groq/compound returns 413 on these prompts, and Agno drops
    gpt-oss content. Direct Groq chat completions avoids both issues.
    """
    model = getattr(agent, "model", None)
    if isinstance(model, Groq):
        api_key = (
            getattr(model, "api_key", None)
            or os.getenv("GROK_API_KEY")
            or os.getenv("GROQ_API_KEY")
        )
        instructions = getattr(agent, "instructions", None) or []
        if isinstance(instructions, str):
            instruction_text = instructions
        else:
            instruction_text = "\n".join(str(item) for item in instructions)
        description = str(getattr(agent, "description", "") or "")
        system_prompt = f"{description}\n{instruction_text}".strip() or "You are a concise travel assistant."
        client = GroqSDK(api_key=api_key)
        create_kwargs = {
            "model": "openai/gpt-oss-20b",
            "messages": [
                {"role": "system", "content": system_prompt[:1200]},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 1800,
            "temperature": 0.4,
        }
        try:
            completion = client.chat.completions.create(
                **create_kwargs,
                reasoning_format="hidden",
                reasoning_effort="low",
            )
        except Exception:
            completion = client.chat.completions.create(**create_kwargs)
        message = completion.choices[0].message
        text = (message.content or "").strip()
        return SimpleNamespace(content=text)
    return agent.run(prompt, stream=False)


def truncate_content(content: str, max_length: int = 2500) -> str:
    """Truncate content to avoid token limit issues"""
    if not content or len(content) <= max_length:
        return content
    return content[:max_length] + "\n\n[Content truncated due to length...]"


def format_weather_brief(weather_data) -> str:
    """Compact weather into a few lines so model prompts stay small."""
    if not weather_data:
        return ""
    lines = []
    for day in weather_data[:14]:
        place = day.get("location") or ""
        prefix = f"{place} " if place else ""
        rain = day.get("precipitation_prob")
        rain_txt = f"{rain}%" if rain is not None else "n/a"
        lines.append(
            f"{prefix}{day.get('date')}: {day.get('status')}, "
            f"{day.get('temp_min')}–{day.get('temp_max')}°C, rain {rain_txt}"
        )
    return "\n".join(lines)


def weather_rows_for_display(weather_data):
    rows = []
    for day in weather_data or []:
        rows.append({
            "Place": day.get("location") or "",
            "Date": day.get("date") or "",
            "Conditions": day.get("status") or "",
            "Low °C": day.get("temp_min"),
            "High °C": day.get("temp_max"),
            "Rain %": day.get("precipitation_prob"),
        })
    return rows


def split_recommended_alternative(text: str):
    """Split a model reply into recommended vs alternative blocks."""
    if not text:
        return "", ""
    parts = re.split(r"(?im)\n+\s*#{0,3}\s*alternative\b[:\s]*", text, maxsplit=1)
    if len(parts) == 2:
        recommended = re.sub(r"(?im)^#{0,3}\s*recommended\b[:\s]*", "", parts[0]).strip()
        alternative = parts[1].strip()
        return recommended, alternative
    return text.strip(), ""


def uniquify_hotel_blocks(recommended: str, alternative: str):
    """Drop duplicate hotel names within and across Recommended / Alternative."""
    seen = set()

    def filter_block(block):
        lines_out = []
        for line in (block or "").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"(?i)^#{0,3}\s*(recommended|alternative)\b", stripped):
                continue
            key = normalize_hotel_key(stripped)
            if not key or key in seen:
                continue
            seen.add(key)
            if not stripped.startswith("-"):
                stripped = f"- {stripped}"
            lines_out.append(stripped)
        return "\n".join(lines_out)

    return filter_block(recommended), filter_block(alternative)


def fill_alt_from_hits(hotel_hits, recommended_text, currency, nightly_min=None, nightly_max=None):
    """Build Alternative stays from leftover unique search hits."""
    used = {
        normalize_hotel_key(line)
        for line in (recommended_text or "").splitlines()
        if line.strip()
    }
    alt_lines = []
    ccy = currency or "INR"
    range_note = ""
    if nightly_min is not None and nightly_max is not None:
        range_note = f" Aim for about {ccy} {int(nightly_min)}–{int(nightly_max)} per night."
    for item in hotel_hits or []:
        city = item.get("city") or "City"
        added_for_city = 0
        for hit in item.get("hits") or []:
            key = normalize_hotel_key(hit.get("title") or "")
            if not key or key in used:
                continue
            used.add(key)
            alt_lines.append(_format_hotel_bullet(city, hit, ccy, "backup stay") + range_note)
            added_for_city += 1
            if added_for_city >= 3:
                break
    return "\n".join(alt_lines)


def looks_like_reasoning(text: str) -> bool:
    """Catch gpt-oss chain-of-thought that leaked into the visible answer."""
    if not text:
        return True
    lowered = text.lower()
    markers = (
        "we need to output",
        "the instruction:",
        "developer instruction",
        "system > developer",
        "which to follow",
        "hierarchy:",
        "let's pick",
        "thinking process",
        "the user explicitly",
        "overrides user",
        "there is a conflict",
        "so we should provide",
        "usually developer instructions",
        "no extra commentary",
        "use headings exactly",
    )
    hits = sum(1 for marker in markers if marker in lowered)
    return hits >= 2


def strip_reasoning(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<think>.*", "", text, flags=re.DOTALL | re.IGNORECASE)
    if looks_like_reasoning(text):
        match = re.search(r"(?im)^(?:#{0,3}\s*)?(recommended|day\s*1)\b", text)
        if match and match.start() > 60:
            text = text[match.start():]
    return text.strip()


def extract_agent_text(result) -> str:
    """Get clean text from an agent result and drop model error payloads."""
    if result is None:
        return ""
    text = getattr(result, "content", None)
    if text is None:
        text = str(result)
    return strip_reasoning(str(text).strip())


def is_failed_model_output(text: str) -> bool:
    if not text:
        return True
    if looks_like_reasoning(text):
        return True
    lowered = text.lower()
    failure_markers = (
        "request_too_large",
        "request entity too large",
        "invalid_request_error",
        '"error":{"message"',
    )
    return any(marker in lowered for marker in failure_markers)


def has_hotel_substance(text: str) -> bool:
    if not text or is_failed_model_output(text):
        return False
    return bool(re.search(r"(?i)(hotel|inn|resort|hostel|palace|haveli|booking\.com|hotels\.com)", text))


def create_llm(api_key: str):
    """Pick a provider from the API key prefix.

    gsk_  -> Groq chat model (compound models reject these prompts as too large)
    xai-  -> xAI Grok
    sk-or- -> OpenRouter
    otherwise -> OpenAI
    """
    if api_key.startswith("gsk_"):
        return Groq(id="openai/gpt-oss-20b", api_key=api_key, max_tokens=1800)
    if api_key.startswith("xai-"):
        return xAI(id="grok-4-1-fast-non-reasoning-latest", api_key=api_key)
    if api_key.startswith("sk-or-"):
        return OpenRouter(id="openai/gpt-4o", api_key=api_key, max_tokens=2048)
    return OpenAIChat(id="gpt-4o", api_key=api_key)


def get_server_api_key() -> str:
    """Load the LLM key from Streamlit secrets or the environment. Never send it to the browser."""
    names = ("GROK_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY")
    for name in names:
        try:
            value = st.secrets.get(name)
            if value:
                return str(value).strip()
        except Exception:
            pass
        value = os.getenv(name)
        if value:
            return value.strip()
    return ""


def get_weather_info(destination: str, start_date: datetime, num_days: int):
    """Get weather forecast for destination using Open-Meteo (free, no API key required)"""
    debug_log("travel_agent.py:19", "get_weather_info called", {"destination": destination, "num_days": num_days}, "B")
    try:
        geocode_url = "https://geocoding-api.open-meteo.com/v1/search"
        geocode_params = {
            "name": destination.split(",")[0].strip() or destination,
            "count": 1,
            "language": "en",
            "format": "json"
        }
        geocode_response = requests.get(geocode_url, params=geocode_params, timeout=10)
        if geocode_response.status_code != 200:
            return None

        geocode_data = geocode_response.json()
        if not geocode_data.get("results"):
            return None

        location = geocode_data["results"][0]
        latitude = location["latitude"]
        longitude = location["longitude"]
        place_name = location.get("name") or destination

        if isinstance(start_date, datetime):
            start = start_date.date()
        else:
            start = start_date
        today = datetime.today().date()
        end = start + timedelta(days=max(num_days - 1, 0))
        forecast_limit = today + timedelta(days=15)

        weather_url = "https://api.open-meteo.com/v1/forecast"
        weather_params = {
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,temperature_2m_min,weather_code,precipitation_probability_max",
            "timezone": "auto",
        }
        if end < today:
            weather_params["forecast_days"] = min(16, max(num_days, 1))
        else:
            req_start = max(start, today)
            req_end = min(end, forecast_limit)
            if req_start > req_end:
                weather_params["forecast_days"] = min(16, max(num_days, 1))
            else:
                weather_params["start_date"] = req_start.strftime("%Y-%m-%d")
                weather_params["end_date"] = req_end.strftime("%Y-%m-%d")

        weather_response = requests.get(weather_url, params=weather_params, timeout=10)
        if weather_response.status_code != 200:
            weather_params.pop("start_date", None)
            weather_params.pop("end_date", None)
            weather_params["forecast_days"] = min(16, max(num_days, 1))
            weather_response = requests.get(weather_url, params=weather_params, timeout=10)
            if weather_response.status_code != 200:
                return None

        weather_data = weather_response.json()
        daily_data = weather_data.get("daily", {})
        weather_codes = {
            0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
            45: "Foggy", 48: "Depositing rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
            55: "Dense drizzle", 56: "Light freezing drizzle", 57: "Dense freezing drizzle",
            61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
            71: "Slight snow fall", 73: "Moderate snow fall", 75: "Heavy snow fall",
            77: "Snow grains", 80: "Slight rain showers", 81: "Moderate rain showers",
            82: "Violent rain showers", 85: "Slight snow showers", 86: "Heavy snow showers",
            95: "Thunderstorm", 96: "Thunderstorm with slight hail", 99: "Thunderstorm with heavy hail"
        }

        weather_info = []
        dates = daily_data.get("time", [])
        temps_max = daily_data.get("temperature_2m_max", [])
        temps_min = daily_data.get("temperature_2m_min", [])
        codes = daily_data.get("weather_code") or daily_data.get("weathercode") or []
        precipitation = daily_data.get("precipitation_probability_max") or daily_data.get("precipitation_probability") or []

        for i in range(min(len(dates), max(num_days, 1))):
            code = codes[i] if i < len(codes) else 0
            status = weather_codes.get(code, "Unknown")
            weather_info.append({
                "date": dates[i],
                "temp_max": temps_max[i] if i < len(temps_max) else None,
                "temp_min": temps_min[i] if i < len(temps_min) else None,
                "status": status,
                "precipitation_prob": precipitation[i] if i < len(precipitation) else None,
                "location": place_name,
            })
        return weather_info or None
    except Exception as e:
        debug_log("travel_agent.py:100", "get_weather_info exception", {"error": str(e)}, "B")
        return None


def generate_ics_content(plan_text: str, start_date: datetime = None) -> bytes:
    """Generate an ICS calendar file from a travel itinerary text."""
    cal = Calendar()
    cal.add('prodid', '-//AI Travel Planner//github.com//')
    cal.add('version', '2.0')

    if start_date is None:
        start_date = datetime.today()
    
    # Convert to date if it's a datetime object
    if isinstance(start_date, datetime):
        start_date_obj = start_date
        start_date = start_date.date()
    else:
        start_date_obj = datetime.combine(start_date, datetime.min.time())

    day_pattern = re.compile(r'Day (\d+)(?:\s*\(([^)]+)\))?[:\s]+(.*?)(?=Day \d+|$)', re.DOTALL)
    days = day_pattern.findall(plan_text)

    if not days:
        event = Event()
        event.add('summary', "Travel Itinerary")
        event.add('description', plan_text)
        event.add('dtstart', start_date)
        event.add('dtend', start_date)
        event.add("dtstamp", datetime.now())
        cal.add_component(event)
    else:
        for day_num, city_name, day_content in days:
            day_num = int(day_num)
            current_date = start_date + timedelta(days=day_num - 1)
            title = f"Day {day_num} ({city_name}) Itinerary" if city_name else f"Day {day_num} Itinerary"

            event = Event()
            event.add('summary', title)
            event.add('description', day_content.strip())
            event.add('dtstart', current_date)
            event.add('dtend', current_date)
            event.add("dtstamp", datetime.now())
            cal.add_component(event)

    return cal.to_ical()


# Streamlit App


# Initialize session state
if 'itinerary' not in st.session_state:
    st.session_state.itinerary = None
if 'weather_info' not in st.session_state:
    st.session_state.weather_info = None
if 'activities_info' not in st.session_state:
    st.session_state.activities_info = None
if 'hotels_info' not in st.session_state:
    st.session_state.hotels_info = None
if 'budget_info' not in st.session_state:
    st.session_state.budget_info = None
if 'city_plan_text' not in st.session_state:
    st.session_state.city_plan_text = None
if 'destination' not in st.session_state:
    st.session_state.destination = None
if 'weather_rows' not in st.session_state:
    st.session_state.weather_rows = None
if 'weather_packing' not in st.session_state:
    st.session_state.weather_packing = None
if 'itinerary_recommended' not in st.session_state:
    st.session_state.itinerary_recommended = None
if 'itinerary_alternative' not in st.session_state:
    st.session_state.itinerary_alternative = None
if 'hotels_recommended' not in st.session_state:
    st.session_state.hotels_recommended = None
if 'hotels_alternative' not in st.session_state:
    st.session_state.hotels_alternative = None
if 'weather_enabled' not in st.session_state:
    st.session_state.weather_enabled = False
if 'display_currency' not in st.session_state:
    st.session_state.display_currency = "INR"
if 'detected_currency' not in st.session_state:
    st.session_state.detected_currency = "INR"
if 'currency_manual' not in st.session_state:
    st.session_state.currency_manual = False
if 'budget_data' not in st.session_state:
    st.session_state.budget_data = None
if 'price_currency' not in st.session_state:
    st.session_state.price_currency = "INR"

# Old sessions defaulted to USD. Keep INR unless the traveler picked a currency.
if not st.session_state.currency_manual and st.session_state.get("display_currency") == "USD":
    st.session_state.display_currency = "INR"

if st.session_state.pop("_use_destination_ccy", False):
    dest_ccy = st.session_state.get("detected_currency")
    if dest_ccy:
        st.session_state.display_currency = dest_ccy
        st.session_state.currency_manual = True

st.markdown(
    """
    <style>
    div[data-testid="stPopover"] > div > button {
        border-radius: 999px;
        white-space: nowrap;
        min-height: 2.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

title_col, currency_col = st.columns([5, 1.35])
with title_col:
    st.title("🧭 Pocket Route")
    st.caption("Plan single-city or multi-city trips with hotels, budget estimates, weather, and booking links")
with currency_col:
    current_ccy = st.session_state.get("display_currency", "INR")
    currency_choices = list(CURRENCY_OPTIONS)
    detected_ccy = st.session_state.get("detected_currency")
    if detected_ccy and detected_ccy not in currency_choices:
        currency_choices = [detected_ccy] + currency_choices
    with st.popover(f"💱 {current_ccy}", help="Change display currency", use_container_width=True):
        st.selectbox(
            "Show prices in",
            currency_choices,
            key="display_currency",
            on_change=mark_currency_manual,
            help="Defaults to INR. Change it anytime.",
        )
        detected = st.session_state.get("detected_currency")
        if detected:
            st.caption(f"Destination currency: {detected}")
            if st.button("Use destination currency", use_container_width=True):
                st.session_state._use_destination_ccy = True
                st.rerun()

openai_api_key = get_server_api_key()

st.sidebar.header("⚙️ Options")
use_weather = st.sidebar.checkbox("Enable Weather Forecast", value=True, 
                                   help="Free weather data using Open-Meteo API (no key required)")
use_activities = st.sidebar.checkbox("Enable Activity Booking Links", value=True)
use_hotels = st.sidebar.checkbox("Enable Hotel Suggestions", value=True)
use_budget = st.sidebar.checkbox("Enable Budget Estimate", value=True)

# Travel Preferences
st.sidebar.header("✈️ Travel Preferences")
travel_style = st.sidebar.selectbox("Travel Style", 
    ["Adventure", "Relaxation", "Culture", "Family-Friendly", "Business", "Luxury"])
budget_range = st.sidebar.selectbox("Budget Range",
    ["Budget-Friendly", "Mid-Range", "Luxury"],
    help="A starting band. Set the nightly stay range below so hotels match what that means for you.")
pref_ccy = st.session_state.get("display_currency") or "INR"
default_lo, default_hi = nightly_defaults_local(budget_range, pref_ccy)
range_sig = f"{budget_range}|{pref_ccy}"
if st.session_state.get("nightly_range_sig") != range_sig:
    st.session_state.nightly_range_sig = range_sig
    st.session_state.nightly_min = default_lo
    st.session_state.nightly_max = default_hi
st.sidebar.markdown(f"**Your {budget_range.lower()} nightly stay** ({pref_ccy} / room)")
step = 500 if pref_ccy in {"INR", "JPY", "KRW", "VND", "IDR", "PKR", "NPR"} else 10
nightly_min = st.sidebar.number_input(
    "From", min_value=1, step=step, key="nightly_min",
    help="Lowest nightly room rate you want hotel suggestions to use.",
)
nightly_max = st.sidebar.number_input(
    "To", min_value=1, step=step, key="nightly_max",
    help="Highest nightly room rate you want hotel suggestions to use.",
)
if nightly_max < nightly_min:
    nightly_max = nightly_min
st.sidebar.caption("Hotels and the lodging line in the budget estimate stay inside this range.")
start_date = st.sidebar.date_input("Travel Start Date", value=datetime.today())
travelers = st.sidebar.number_input("Travelers", min_value=1, max_value=8, value=1)
interests = st.sidebar.multiselect("Interests",
    ["Food & Dining", "History & Culture", "Nature & Outdoors", 
     "Nightlife", "Shopping", "Art & Museums", "Sports"])

if openai_api_key:
    # Existing Agents
    researcher = Agent(
        name="Researcher",
        role="Searches for travel destinations, activities, and accommodations",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a world-class travel researcher. Generate targeted search terms 
            and find relevant travel information, activities, and accommodations.
        """),
        instructions=[
            "Use the provided DuckDuckGo research summary plus your knowledge.",
            "Organize results by: Attractions, Restaurants, Accommodations, Activities, Tips.",
            "Include approximate costs when mentioned.",
            "Prioritize recent and highly-rated options.",
            "Keep your response concise - summarize key information, don't include full article text.",
            "Focus on top 3-5 results per category to avoid overwhelming responses.",
        ],
        tools=[],
    )
    
    planner = Agent(
        name="Planner",
        role="Generates detailed day-by-day itineraries",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a senior travel planner. Create detailed itineraries with time slots, 
            costs, and practical information.
        """),
        instructions=[
            "Create day-by-day itinerary with time slots (Morning, Afternoon, Evening).",
            "Follow the city day allocation exactly when multiple cities are provided.",
            "Use headings like 'Day 1 (City):' so each day is tied to one city.",
            "Include a short transfer note on days you move between cities.",
            "Include estimated costs for each activity when available.",
            "Suggest realistic travel times between locations.",
            "Balance activities - don't overpack days.",
            "Include meal recommendations with cuisine types and price ranges.",
            "Add practical tips: best times to visit, booking requirements.",
            "List specific activity names clearly for booking purposes.",
            "Never make up facts - only use research results.",
        ],
    )
    
    # NEW AGENT 1: Weather Agent
    weather_agent = Agent(
        name="WeatherAnalyst",
        role="Provides weather forecasts and packing suggestions",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a weather and travel preparation expert. Analyze weather forecasts 
            and provide practical packing and activity recommendations.
        """),
        instructions=[
            "Analyze weather data provided for the destination and dates.",
            "Provide daily weather summaries with temperature and conditions.",
            "Suggest appropriate clothing and packing items.",
            "Recommend indoor alternatives if weather is unfavorable.",
            "Give practical tips based on weather patterns.",
            "Format weather info clearly by day.",
        ],
    ) if use_weather else None
    
    # NEW AGENT 2: Activities Agent (FOCUSED ON BOOKING LINKS)
    activities_agent = Agent(
        name="ActivitiesFinder",
        role="Identifies activities and finds booking links",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are an activities and booking specialist. Identify specific activities 
            from itineraries and suggest where to book (official sites or major platforms).
        """),
        instructions=[
            "Extract 3-5 key activities and attractions mentioned in the itinerary.",
            "Suggest likely booking platforms or official sites; include plausible URLs if known, otherwise name the platform.",
            "Organize results by: Attractions, Restaurants, Accommodations, Activities, Tips.",
            "Include approximate costs when mentioned.",
            "Prioritize recent and highly-rated options.",
            "Keep your response concise - summarize key information, don't include full article text.",
            "Focus on top 3-5 results per category to avoid overwhelming responses.",
        ],
        tools=[],
    )
    
    # NEW AGENT 3: Logistics Agent
    logistics_agent = Agent(
        name="LogisticsPlanner",
        role="Plans transportation and routes between locations",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a logistics and transportation expert. Plan optimal routes, 
            suggest transportation modes, and estimate travel times.
        """),
        instructions=[
            "Analyze the itinerary and identify locations to visit.",
            "Suggest optimal routes and transportation modes (walking, public transit, taxi, rental car).",
            "Estimate travel times between activities.",
            "Recommend efficient day plans to minimize travel time.",
            "Provide practical transportation tips.",
            "Format logistics info clearly with routes and times.",
        ],
        tools=[],
    )

    hotel_agent = Agent(
        name="HotelFinder",
        role="Suggests hotels that match budget, dates, and city stays",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a hotel specialist. Recommend several in-budget hotels or
            guesthouses for each city. Never suggest luxury palaces when the
            traveler set a modest nightly range.
        """),
        instructions=[
            "For each city, list 3 Recommended stays and 3 Alternative stays.",
            "Use the headings Recommended and Alternative on their own lines.",
            "Never repeat a hotel. Alternative must be different properties from Recommended, and each name may appear only once.",
            "Every stay MUST fit the traveler's nightly min–max in the requested currency. Skip 5-star, palace, and luxury properties when the max is a budget or mid-range amount.",
            "Each stay: hotel name, neighborhood, nightly price in the requested currency, why it fits, booking site.",
            "Prefer search-note properties that look affordable. If notes are thin, name well-known budget or mid-range stays, not landmark luxury hotels.",
            "Output the hotel list only. Never discuss instructions, conflicts, or your reasoning.",
        ],
        tools=[],
    ) if use_hotels else None

    budget_agent = Agent(
        name="BudgetEstimator",
        role="Turns a trip plan into a clear cost breakdown",
        model=create_llm(openai_api_key),
        description=dedent("""\
            You are a travel budget analyst. Refine a numeric baseline into a
            readable estimate travelers can use for planning.
        """),
        instructions=[
            "Keep the baseline totals unless a city is clearly much cheaper or more expensive.",
            "Break costs into lodging, food, activities, local transport, and intercity travel.",
            "Call out what is included vs not (flights from home, insurance, shopping).",
            "Stay under 180 words. Use bullet points and the requested currency.",
        ],
        tools=[],
    ) if use_budget else None

    # Main Input Section
    st.header("🎯 Plan Your Trip")
    destination_input = st.text_area(
        "📍 Cities to visit",
        placeholder="Paris, France\nRome, Italy",
        help="One city, or up to 4 cities in visit order. Put each city on its own line, e.g. Paris, France then Rome, Italy.",
        height=90,
    )
    col1, col2 = st.columns(2)
    with col1:
        num_days = st.number_input("📅 Number of Days", min_value=1, max_value=30, value=7)
    with col2:
        st.caption("Days are split across cities in the order you listed them.")

    if st.button("🚀 Generate Complete Itinerary", type="primary", use_container_width=True):
        cities = parse_cities(destination_input)
        city_plan = allocate_city_days(cities, num_days)
        destination = destination_label(city_plan)
        city_plan_text = format_city_plan(city_plan)
        inferred_currency = infer_currency_from_places([stop["city"] for stop in city_plan])
        trip_currency = st.session_state.get("display_currency") or "INR"
        currency_symbol = CURRENCY_SYMBOLS.get(trip_currency, f"{trip_currency} ")
        range_ccy = trip_currency
        nightly_min_local = min(int(nightly_min), int(nightly_max))
        nightly_max_local = max(int(nightly_min), int(nightly_max))
        nightly_min_trip = nightly_min_local
        nightly_max_trip = nightly_max_local
        lodging_night_usd = (
            currency_to_usd(nightly_min_local, range_ccy),
            currency_to_usd(nightly_max_local, range_ccy),
        )
        debug_log("travel_agent.py:292", "Button clicked", {"destination": destination, "num_days": num_days, "currency": trip_currency, "nightly": [nightly_min_trip, nightly_max_trip]}, "A")
        if not city_plan:
            st.error("Please enter at least one city.")
        else:
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Research (15%)
                # #region agent log
                debug_log("travel_agent.py:303", "Starting Step 1: Research", {}, "A")
                # #endregion
                status_text.text("🔍 Researching destination...")
                progress_bar.progress(15)
                start_time = time.time()
                with st.spinner("🔍 Searching for travel information... This may take ~30 seconds."):
                    research_text = build_multi_city_research(city_plan, travel_style, budget_range, interests)
                # #region agent log
                elapsed = time.time() - start_time
                debug_log("travel_agent.py:315", "After duckduckgo research", {"elapsed_seconds": elapsed, "has_content": bool(research_text)}, "A")
                # #endregion
                
                # Step 2: Weather Analysis (30%)
                # #region agent log
                debug_log("travel_agent.py:320", "Starting Step 2: Weather Analysis", {"use_weather": use_weather}, "B")
                # #endregion
                weather_data = None
                weather_analysis = None
                if use_weather:
                    status_text.text("🌤️ Getting weather forecast...")
                    progress_bar.progress(30)
                    weather_data = []
                    for stop in city_plan:
                        city_start = start_date + timedelta(days=stop["start_day"] - 1)
                        city_weather = get_weather_info(stop["city"], city_start, stop["days"])
                        if city_weather:
                            for day in city_weather:
                                day["location"] = stop["city"]
                            weather_data.extend(city_weather)
                    if weather_data and weather_agent:
                        weather_prompt = f"""
                        Cities: {destination}
                        City plan:
                        {city_plan_text}
                        Weather Data:
                        {format_weather_brief(weather_data)}

                        Provide a short weather analysis and packing recommendations for each city.
                        Keep the answer under 200 words.
                        """
                        # #region agent log
                        debug_log("travel_agent.py:331", "Before weather_agent.run()", {}, "B")
                        start_time = time.time()
                        # #endregion
                        try:
                            weather_analysis = run_agent_with_timeout(weather_agent, weather_prompt, timeout_seconds=120, agent_name="WeatherAgent")
                        except Exception as weather_error:
                            debug_log("travel_agent.py:332", "weather_agent failed", {"error": str(weather_error)}, "B")
                            weather_analysis = None
                        # #region agent log
                        elapsed = time.time() - start_time
                        debug_log("travel_agent.py:332", "After weather_agent.run()", {"elapsed_seconds": elapsed}, "B")
                        # #endregion
                
                # Step 3: Logistics Planning (45%)
                # #region agent log
                debug_log("travel_agent.py:340", "Starting Step 3: Logistics Planning", {}, "C")
                # #endregion
                status_text.text("🗺️ Planning routes and transportation...")
                progress_bar.progress(45)
                logistics_research = truncate_content(research_text, max_length=900)
                logistics_prompt = f"""
                Trip: {destination}
                Duration: {num_days} days
                City plan:
                {city_plan_text}

                Research Summary:
                {logistics_research}

                Suggest local transport plus how to travel between cities if there is more than one.
                Keep the answer under 200 words.
                """
                # #region agent log
                debug_log("travel_agent.py:353", "Before logistics_agent.run()", {"prompt_length": len(logistics_prompt)}, "C")
                start_time = time.time()
                # #endregion
                try:
                    logistics_results = run_agent_with_timeout(logistics_agent, logistics_prompt, timeout_seconds=120, agent_name="LogisticsAgent")
                except Exception as logistics_error:
                    debug_log("travel_agent.py:354", "logistics_agent failed", {"error": str(logistics_error)}, "C")
                    logistics_results = None
                # #region agent log
                elapsed = time.time() - start_time
                debug_log("travel_agent.py:354", "After logistics_agent.run()", {"elapsed_seconds": elapsed}, "C")
                # #endregion
                
                # Step 4: Create Itinerary (60%)
                # #region agent log
                debug_log("travel_agent.py:360", "Starting Step 4: Create Itinerary", {}, "D")
                # #endregion
                status_text.text("📋 Creating your personalized itinerary...")
                progress_bar.progress(60)
                
                # Build weather and logistics info strings separately to avoid f-string backslash issue
                newline = "\n"
                weather_section = ""
                weather_text = extract_agent_text(weather_analysis)
                if weather_text and not is_failed_model_output(weather_text):
                    weather_content = truncate_content(weather_text, max_length=500)
                    weather_section = f"Weather Info:{newline}{weather_content}"
                elif weather_data:
                    weather_section = f"Weather Info:{newline}{format_weather_brief(weather_data)}"

                logistics_section = ""
                logistics_text = extract_agent_text(logistics_results)
                if logistics_text and not is_failed_model_output(logistics_text):
                    logistics_content = truncate_content(logistics_text, max_length=500)
                    logistics_section = f"Logistics Info:{newline}{logistics_content}"

                research_content = truncate_content(research_text, max_length=1400)
                
                planning_prompt = f"""
                Trip: {destination}
                Duration: {num_days} days
                Travel Style: {travel_style}
                Budget Range: {budget_range}
                Nightly hotel budget: {int(nightly_min_trip)}–{int(nightly_max_trip)} {trip_currency} per room
                Currency for all prices: {trip_currency} ({currency_symbol.strip()})
                Travelers: {travelers}
                Interests: {', '.join(interests) if interests else 'General'}
                Start Date: {start_date.strftime('%B %d, %Y')}

                City day allocation (follow exactly):
                {city_plan_text}

                Research Results:
                {research_content}

                {weather_section}
                {logistics_section}

                Create a concise day-by-day itinerary with:
                - Headings like "Day 1 (City):"
                - Time slots (Morning: 9-12, Afternoon: 12-5, Evening: 5-9)
                - Activity descriptions and approximate costs in {trip_currency} only
                - A short transfer note when moving to the next city
                - Meal recommendations

                Keep the itinerary focused and under 800 words.
                """
                # #region agent log
                debug_log("travel_agent.py:404", "Before planner.run()", {"prompt_length": len(planning_prompt)}, "D")
                start_time = time.time()
                # #endregion
                response = run_agent_with_timeout(planner, planning_prompt, timeout_seconds=120, agent_name="Planner")
                # #region agent log
                elapsed = time.time() - start_time
                debug_log("travel_agent.py:405", "After planner.run()", {"elapsed_seconds": elapsed}, "D")
                # #endregion
                progress_bar.progress(75)

                itinerary_text = extract_agent_text(response)
                if is_failed_model_output(itinerary_text):
                    raise RuntimeError(
                        "The model could not generate an itinerary because the request was too large. "
                        "Please try again with a shorter trip."
                    )

                alternative_itinerary = None
                status_text.text("🔀 Drafting an alternative itinerary...")
                progress_bar.progress(72)
                alt_prompt = f"""
                Create a DIFFERENT itinerary for the same trip. Do not copy the first plan's main sights.
                Same cities, days, budget ({budget_range}, hotels {int(nightly_min_trip)}–{int(nightly_max_trip)} {trip_currency}/night), and style ({travel_style}).
                Keep all approximate costs in {trip_currency} ({currency_symbol.strip()}) only.
                Use headings like "Day 1 (City):". Under 800 words.

                City plan:
                {city_plan_text}

                First plan (avoid repeating it):
                {truncate_content(itinerary_text, 700)}
                """
                try:
                    alt_response = run_agent_with_timeout(planner, alt_prompt, timeout_seconds=120, agent_name="PlannerAlternative")
                    alternative_itinerary = extract_agent_text(alt_response)
                    if is_failed_model_output(alternative_itinerary):
                        alternative_itinerary = None
                except Exception as alt_error:
                    debug_log("travel_agent.py:alt_itinerary", "alternative itinerary failed", {"error": str(alt_error)}, "D")
                    alternative_itinerary = None

                # Step 5: Find Activity Booking Links (90%)
                # #region agent log
                debug_log("travel_agent.py:410", "Starting Step 5: Activity Booking Links", {"use_activities": use_activities}, "E")
                # #endregion
                activities_info = None
                if use_activities and activities_agent:
                    status_text.text("🎫 Finding activity booking links...")
                    progress_bar.progress(75)

                    itinerary_content = truncate_content(itinerary_text, max_length=900)

                    activities_prompt = f"""
                    From this itinerary, list 3-5 real attraction names with likely booking links.
                    Do not mention API errors. If names are missing, use well-known attractions in {destination}.

                    Itinerary:
                    {itinerary_content}

                    Prefer Viator, GetYourGuide, TripAdvisor, Klook, or official sites.
                    Return a short bullet list only.
                    """
                    # #region agent log
                    debug_log("travel_agent.py:433", "Before activities_agent.run()", {"prompt_length": len(activities_prompt)}, "E")
                    start_time = time.time()
                    # #endregion
                    try:
                        activities_info = run_agent_with_timeout(activities_agent, activities_prompt, timeout_seconds=120, agent_name="ActivitiesAgent")
                    except Exception as activities_error:
                        debug_log("travel_agent.py:434", "activities_agent failed", {"error": str(activities_error)}, "E")
                        activities_info = None
                    # #region agent log
                    elapsed = time.time() - start_time
                    debug_log("travel_agent.py:434", "After activities_agent.run()", {"elapsed_seconds": elapsed}, "E")
                    # #endregion

                hotels_info = None
                hotel_hits = []
                if use_hotels and hotel_agent:
                    status_text.text("🏨 Finding hotel options...")
                    progress_bar.progress(88)
                    hotel_hits = collect_hotel_hits(
                        city_plan, budget_range, nightly_min_trip, nightly_max_trip, trip_currency
                    )
                    hotel_notes = truncate_content(format_hotel_research_notes(hotel_hits), max_length=1200)
                    hotels_prompt = f"""
                    Output ONLY this structure. Do not explain, debate instructions, or think out loud.

                    Recommended
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.

                    Alternative
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.
                    - City: Hotel name. Neighborhood. Nightly {trip_currency} price. Why it fits. Booking site.

                    Budget band: {budget_range}
                    Nightly room budget: {nightly_min_trip}–{nightly_max_trip} {trip_currency} ONLY. Do not suggest hotels above {nightly_max_trip} {trip_currency}.
                    Travelers: {travelers}
                    Stay plan:
                    {city_plan_text}

                    Hotel search notes:
                    {hotel_notes}

                    Three recommended stays and three alternative stays per city, all inside that nightly range.
                    Never repeat a hotel name. Alternative must be different properties from Recommended.
                    Short bullets only.
                    """
                    try:
                        hotels_info = run_agent_with_timeout(hotel_agent, hotels_prompt, timeout_seconds=120, agent_name="HotelAgent")
                    except Exception as hotel_error:
                        debug_log("travel_agent.py:hotels", "hotel_agent failed", {"error": str(hotel_error)}, "F")
                        hotels_info = None

                budget_info = None
                budget_data = estimate_trip_budget(
                    budget_range, num_days, len(city_plan), travelers, lodging_night_usd
                )
                baseline_budget = format_budget_estimate(budget_data, trip_currency)
                if use_budget:
                    status_text.text("💰 Estimating budget...")
                    progress_bar.progress(94)
                    if budget_agent:
                        budget_prompt = f"""
                        Refine this baseline budget for {destination}.
                        Travelers: {travelers}
                        Stay plan:
                        {city_plan_text}

                        Baseline:
                        {baseline_budget}

                        Hotel notes:
                        {truncate_content(extract_agent_text(hotels_info), 400) if hotels_info else "None"}

                        Keep all amounts in {trip_currency} ({currency_symbol.strip()}). Lodging should follow {nightly_min_trip}–{nightly_max_trip} {trip_currency} per room per night. Mention that flights from home are not included.
                        Under 180 words.
                        """
                        try:
                            budget_info = run_agent_with_timeout(budget_agent, budget_prompt, timeout_seconds=120, agent_name="BudgetAgent")
                        except Exception as budget_error:
                            debug_log("travel_agent.py:budget", "budget_agent failed", {"error": str(budget_error)}, "G")
                            budget_info = None
                
                progress_bar.progress(100)
                status_text.text("✅ Itinerary complete!")

                activities_text = extract_agent_text(activities_info)
                if is_failed_model_output(activities_text):
                    activities_text = None

                hotels_text = extract_agent_text(hotels_info)
                if not has_hotel_substance(hotels_text):
                    hotels_text = (
                        fallback_hotels_markdown(hotel_hits, trip_currency, nightly_min_trip, nightly_max_trip)
                        if use_hotels else None
                    )
                hotels_recommended, hotels_alternative = split_recommended_alternative(hotels_text)
                hotels_recommended, hotels_alternative = uniquify_hotel_blocks(
                    hotels_recommended, hotels_alternative
                )
                if use_hotels and not (hotels_alternative or "").strip():
                    hotels_alternative = fill_alt_from_hits(
                        hotel_hits, hotels_recommended, trip_currency, nightly_min_trip, nightly_max_trip
                    )
                if hotels_recommended:
                    hotels_text = "Recommended\n" + hotels_recommended
                    if hotels_alternative:
                        hotels_text += "\n\nAlternative\n" + hotels_alternative

                budget_text = extract_agent_text(budget_info)
                if is_failed_model_output(budget_text):
                    budget_text = baseline_budget if use_budget else None
                elif not budget_text and use_budget:
                    budget_text = baseline_budget

                packing_text = weather_text if weather_text and not is_failed_model_output(weather_text) else None
                forecast_text = format_weather_brief(weather_data) if weather_data else None

                st.session_state.itinerary = itinerary_text
                st.session_state.itinerary_recommended = itinerary_text
                st.session_state.itinerary_alternative = alternative_itinerary
                st.session_state.weather_info = packing_text or forecast_text
                st.session_state.weather_packing = packing_text
                st.session_state.weather_rows = weather_rows_for_display(weather_data)
                st.session_state.activities_info = activities_text
                st.session_state.hotels_info = hotels_text
                st.session_state.hotels_recommended = hotels_recommended or hotels_text
                st.session_state.hotels_alternative = hotels_alternative
                st.session_state.budget_info = budget_text if use_budget else None
                st.session_state.budget_data = budget_data if use_budget else None
                st.session_state.detected_currency = inferred_currency
                st.session_state.price_currency = trip_currency
                st.session_state.city_plan_text = city_plan_text
                st.session_state.start_date = start_date
                st.session_state.destination = destination
                st.session_state.plan_choice = "Recommended"
                st.session_state.hotel_choice = "Recommended"
                st.session_state.weather_enabled = bool(use_weather)
                st.session_state.hotel_nightly_min = nightly_min_trip
                st.session_state.hotel_nightly_max = nightly_max_trip
                st.rerun()
            except TimeoutError as e:
                # #region agent log
                debug_log("travel_agent.py:451", "TimeoutError caught", {"error": str(e)}, "A")
                # #endregion
                st.error(f"⏱️ {str(e)}")
                st.warning("💡 **Tips to resolve:**\n- Check your internet connection\n- Verify your API keys are correct and have available credits\n- Try again with a shorter trip duration\n- Disable optional features (weather/activities) to reduce processing time")
            except Exception as e:
                # #region agent log
                debug_log("travel_agent.py:456", "Exception caught", {"error": str(e), "error_type": type(e).__name__}, "A")
                # #endregion
                st.error(f"Error generating itinerary: {str(e)}")
                st.exception(e)

    # Display Results Section
    if st.session_state.itinerary:
        st.divider()
        st.header("📋 Your Itinerary")
        if st.session_state.city_plan_text:
            st.markdown("**City plan**")
            st.markdown(st.session_state.city_plan_text)

        st.subheader("🌤️ Weather forecast")
        weather_rows = st.session_state.get("weather_rows") or []
        if weather_rows:
            st.dataframe(weather_rows, use_container_width=True, hide_index=True)
            packing = st.session_state.get("weather_packing")
            if packing:
                st.markdown(packing)
            st.caption("Daily forecast from Open-Meteo. Open-Meteo covers about the next 16 days; later trip dates use the nearest available forecast.")
        elif st.session_state.get("weather_info"):
            st.markdown(st.session_state.weather_info)
        elif st.session_state.get("weather_enabled"):
            st.info("Weather could not be loaded for this location or date. Try a well-known city name and a start date within the next two weeks.")
        else:
            st.caption("Weather forecast is turned off in the sidebar.")

        plan_options = ["Recommended"]
        if st.session_state.get("itinerary_alternative"):
            plan_options.append("Alternative")
        plan_choice = st.radio(
            "Choose an itinerary",
            plan_options,
            horizontal=True,
            key="plan_choice",
            help="Pick Alternative if you want a different set of sights and pacing.",
        )
        active_itinerary = st.session_state.itinerary_recommended or st.session_state.itinerary
        if plan_choice == "Alternative" and st.session_state.get("itinerary_alternative"):
            active_itinerary = st.session_state.itinerary_alternative
        st.session_state.itinerary = active_itinerary
        display_ccy = st.session_state.get("display_currency") or "INR"
        source_ccy = st.session_state.get("price_currency") or st.session_state.get("detected_currency") or "INR"
        st.caption(f"Plan amounts are in {source_ccy}. Showing {display_ccy}. Change currency with 💱 in the top-right.")
        st.write(text_in_display_currency(active_itinerary))

        if st.session_state.budget_data or st.session_state.budget_info:
            with st.expander("💰 Budget Estimate", expanded=True):
                ccy = st.session_state.get("display_currency") or "INR"
                if st.session_state.budget_data:
                    st.markdown(format_budget_estimate(st.session_state.budget_data, ccy))
                else:
                    st.markdown(text_in_display_currency(st.session_state.budget_info))
                st.caption(f"Showing {ccy}. Change currency with 💱 in the top-right.")

        if st.session_state.hotels_info or st.session_state.get("hotels_recommended"):
            with st.expander("🏨 Hotel Suggestions", expanded=True):
                hotel_options = ["Recommended"]
                if st.session_state.get("hotels_alternative"):
                    hotel_options.append("Alternative")
                hotel_choice = st.radio(
                    "Choose hotel option",
                    hotel_options,
                    horizontal=True,
                    key="hotel_choice",
                    help="Recommended and Alternative are different unique hotels. Switch lists if you want other in-budget stays.",
                )
                hotels_content = st.session_state.get("hotels_recommended") or st.session_state.hotels_info
                if hotel_choice == "Alternative" and st.session_state.get("hotels_alternative"):
                    hotels_content = st.session_state.hotels_alternative
                url_pattern = re.compile(r'(https?://[^\s]+)')
                st.markdown(url_pattern.sub(r'[\1](\1)', text_in_display_currency(hotels_content or "")))
                lo = st.session_state.get("hotel_nightly_min")
                hi = st.session_state.get("hotel_nightly_max")
                ccy = st.session_state.get("display_currency") or "INR"
                src = st.session_state.get("price_currency") or ccy
                if lo and hi:
                    shown_lo = format_local_amount(convert_amount(lo, src, ccy), ccy)
                    shown_hi = format_local_amount(convert_amount(hi, src, ccy), ccy)
                    st.caption(f"Stays aimed at {shown_lo}–{shown_hi} per room / night. Change the sidebar range and generate again to retarget hotels.")
        
        # Display Activities with Booking Links
        if st.session_state.activities_info:
            with st.expander("🎫 Activity Booking Links", expanded=True):
                # Parse and format booking links nicely
                activities_content = st.session_state.activities_info
                
                # Try to extract URLs and format them as clickable links
                url_pattern = re.compile(r'(https?://[^\s]+)')
                activities_formatted = url_pattern.sub(
                    r'[\1](\1)', 
                    activities_content
                )
                st.markdown(activities_formatted)
        
        # Export Options
        st.divider()
        st.header("💾 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            ics_content = generate_ics_content(
                st.session_state.itinerary,
                st.session_state.get('start_date', datetime.today())
            )
            dest_name = st.session_state.destination.replace(' ', '_') if st.session_state.destination else 'travel'
            dest_name = ''.join(c for c in dest_name if c.isalnum() or c in ('_', '-'))  # Sanitize filename
            
            st.download_button(
                label="📅 Calendar (.ics)",
                data=ics_content,
                file_name=f"itinerary_{dest_name}.ics",
                mime="text/calendar",
                use_container_width=True
            )
        
        with col2:
            dest_name = st.session_state.destination.replace(' ', '_') if st.session_state.destination else 'travel'
            dest_name = ''.join(c for c in dest_name if c.isalnum() or c in ('_', '-'))  # Sanitize filename
            
            st.download_button(
                label="📄 Text (.txt)",
                data=text_in_display_currency(st.session_state.itinerary),
                file_name=f"itinerary_{dest_name}.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        with col3:
            dest_name = st.session_state.destination.replace(' ', '_') if st.session_state.destination else 'travel'
            dest_name = ''.join(c for c in dest_name if c.isalnum() or c in ('_', '-'))  # Sanitize filename
            
            # Combined export with weather and activities
            combined_content = f"""
{text_in_display_currency(st.session_state.itinerary)}

{'='*60}
CITY PLAN
{'='*60}
{st.session_state.city_plan_text if st.session_state.city_plan_text else 'Not available'}

{'='*60}
BUDGET ESTIMATE
{'='*60}
{format_budget_estimate(st.session_state.budget_data, st.session_state.get('display_currency') or 'INR') if st.session_state.get('budget_data') else text_in_display_currency(st.session_state.budget_info if st.session_state.budget_info else 'Not available')}

{'='*60}
HOTEL SUGGESTIONS
{'='*60}
{text_in_display_currency(st.session_state.hotels_info if st.session_state.hotels_info else 'Not available')}

{'='*60}
WEATHER FORECAST & PACKING TIPS
{'='*60}
{st.session_state.weather_info if st.session_state.weather_info else 'Not available'}

{'='*60}
ACTIVITY BOOKING LINKS
{'='*60}
{st.session_state.activities_info if st.session_state.activities_info else 'Not available'}
"""
            st.download_button(
                label="📦 Complete Guide",
                data=combined_content,
                file_name=f"complete_guide_{dest_name}.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.error("This app is missing a server API key, so trip planning is unavailable.")
    st.caption("The operator should add GROK_API_KEY (or GROQ_API_KEY / OPENAI_API_KEY) in Streamlit secrets or the environment. Visitors never enter a key.")
