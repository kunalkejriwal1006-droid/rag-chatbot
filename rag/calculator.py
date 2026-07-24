"""Deterministic NCB/IDV calculator, kept separate from the RAG path so
these numbers are never subject to chunking or retrieval errors.

IRDAI Motor Tariff NCB slab table (applies to Own Damage premium only).
Structure: {previous_ncb_pct: {claims_in_expiring_period: new_ncb_pct}}
"""

import re
from typing import Optional

NCB_PROGRESSION = {
    #  prev_ncb -> {0 claims, 1 claim, 2 claims, 3 claims, 4 claims, 5+ claims}
    0:  {0: 20, 1: 0,  2: 0,  3: 0,  4: 0,  5: 0},
    20: {0: 25, 1: 0,  2: 0,  3: 0,  4: 0,  5: 0},
    25: {0: 35, 1: 20, 2: 0,  3: 0,  4: 0,  5: 0},
    35: {0: 45, 1: 25, 2: 0,  3: 0,  4: 0,  5: 0},
    45: {0: 50, 1: 35, 2: 0,  3: 0,  4: 0,  5: 0},
    50: {0: 50, 1: 45, 2: 0,  3: 0,  4: 0,  5: 0},
}

# policy year -> previous NCB% assuming no claims (year 2 = 1st renewal, etc.)
YEAR_TO_PREV_NCB = {1: 0, 2: 0, 3: 20, 4: 25, 5: 35, 6: 45}


def get_ncb_percentage(policy_year: int, claims_in_expiring_period: int = 0) -> int:
    if policy_year <= 1:
        return 0
    prev_ncb = YEAR_TO_PREV_NCB.get(policy_year, 45)
    row = NCB_PROGRESSION.get(prev_ncb, NCB_PROGRESSION[45])
    claim_key = min(claims_in_expiring_period, 5)
    return row.get(claim_key, 0)


def calculate_ncb(
    od_premium: float,
    policy_year: int,
    claims_in_expiring_period: int = 0,
) -> dict:
    """NCB discount applies only to the OD premium component, not TP."""
    prev_ncb = YEAR_TO_PREV_NCB.get(policy_year, 45) if policy_year > 1 else 0
    ncb_pct = get_ncb_percentage(policy_year, claims_in_expiring_period)
    discount = round(od_premium * ncb_pct / 100, 2)
    payable = round(od_premium - discount, 2)

    note = (
        "NCB applies only to the Own Damage (OD) premium component. "
        "Third Party (TP) premium is not discounted by NCB."
    )
    if claims_in_expiring_period > 0 and ncb_pct == 0:
        note += (
            f"  Since {claims_in_expiring_period} claim(s) were made in the "
            "expiring period, NCB resets to 0% — no discount applies."
        )

    return {
        "od_premium_rs": od_premium,
        "policy_year": policy_year,
        "previous_ncb_pct": prev_ncb,
        "ncb_pct_at_renewal": ncb_pct,
        "discount_rs": discount,
        "payable_od_premium_rs": payable,
        "step_by_step": (
            f"Step 1 -> OD Premium = Rs.{od_premium}\n"
            f"Step 2 -> Policy Year = {policy_year} "
            f"(previous NCB = {prev_ncb}%, claims = {claims_in_expiring_period})\n"
            f"Step 3 -> Applicable NCB % = {ncb_pct}%\n"
            f"Step 4 -> NCB Discount = Rs.{od_premium} x {ncb_pct}% = Rs.{discount}\n"
            f"Step 5 -> Payable OD Premium = Rs.{od_premium} - Rs.{discount} = Rs.{payable}"
        ),
        "note": note,
    }


# IRDAI standard depreciation schedule, by completed vehicle age in years.
# 0 = under 6 months.
IDV_DEPRECIATION_TABLE = {
    (0, 0): 5,
    (1, 1): 15,
    (2, 2): 20,
    (3, 3): 30,
    (4, 4): 40,
    (5, 5): 50,
}


def get_depreciation_pct(vehicle_age_years: int) -> Optional[int]:
    """None if > 5 years - depreciation is negotiated at that point."""
    for (lo, hi), pct in IDV_DEPRECIATION_TABLE.items():
        if lo <= vehicle_age_years <= hi:
            return pct
    return None


def calculate_idv(ex_showroom_price: float, vehicle_age_years: int) -> dict:
    """IDV = ex-showroom price x (1 - depreciation%)."""
    dep_pct = get_depreciation_pct(vehicle_age_years)

    if dep_pct is None:
        return {
            "ex_showroom_price_rs": ex_showroom_price,
            "vehicle_age_years": vehicle_age_years,
            "depreciation_pct": "Negotiated",
            "idv_rs": "To be agreed between insurer and insured",
            "step_by_step": (
                f"Vehicle age ({vehicle_age_years} years) exceeds 5 years. "
                "As per IRDAI tariff, IDV for vehicles older than 5 years is "
                "determined by mutual agreement between the insurer and insured."
            ),
            "note": "IDV cannot be auto-calculated for vehicles > 5 years old.",
        }

    dep_rs = round(ex_showroom_price * dep_pct / 100, 2)
    idv = round(ex_showroom_price - dep_rs, 2)

    return {
        "ex_showroom_price_rs": ex_showroom_price,
        "vehicle_age_years": vehicle_age_years,
        "depreciation_pct": dep_pct,
        "depreciation_rs": dep_rs,
        "idv_rs": idv,
        "step_by_step": (
            f"Step 1 -> Ex-Showroom Price = Rs.{ex_showroom_price}\n"
            f"Step 2 -> Vehicle Age = {vehicle_age_years} completed year(s)\n"
            f"Step 3 -> Standard Depreciation % = {dep_pct}%\n"
            f"Step 4 -> Depreciation Amount = Rs.{ex_showroom_price} x {dep_pct}% "
            f"= Rs.{dep_rs}\n"
            f"Step 5 -> IDV = Rs.{ex_showroom_price} - Rs.{dep_rs} = Rs.{idv}"
        ),
        "note": (
            "IDV is the maximum claim amount. It is the ex-showroom price of the "
            "vehicle after deducting standard depreciation. Accessories fitted to "
            "the vehicle are valued separately."
        ),
    }


_NCB_INTENT_RE = re.compile(
    r"\b(ncb|no[\s\-]?claim[\s\-]?bonus)\b", re.IGNORECASE
)
_IDV_INTENT_RE = re.compile(
    r"\b(idv|insured.{0,10}declared[\s\-]?value|declared[\s\-]?value)\b",
    re.IGNORECASE,
)

# handles "5k", "2.5 lakh", "1.2 Cr" style amounts
_AMOUNT_MULTIPLIERS = {
    "k": 1_000, "thousand": 1_000,
    "l": 100_000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
    "cr": 10_000_000, "crore": 10_000_000, "crores": 10_000_000,
}
_NUM_SUFFIX = r"(\d[\d,]*(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?|l|crores?|cr)?"

_PREMIUM_RE = re.compile(
    rf"(?:premium|od|own[\s\-]?damage)\s*(?:is|=|:|of|was)?\s*(?:rs\.?|inr|₹)?\s*{_NUM_SUFFIX}",
    re.IGNORECASE,
)
_PRICE_RE = re.compile(
    rf"(?:price|cost|showroom|ex[\s\-]?showroom)\s*(?:is|=|:|of|was)?\s*(?:rs\.?|inr|₹)?\s*{_NUM_SUFFIX}",
    re.IGNORECASE,
)
_AMOUNT_FALLBACK_RE = re.compile(
    rf"(?:rs\.?|inr|₹)\s*{_NUM_SUFFIX}|{_NUM_SUFFIX}\s*(?:rs\.?|inr|₹)",
    re.IGNORECASE,
)

# "renewal" is offset by one from "year" - 1st renewal = policy year 2
_YEAR_RE = re.compile(
    r"(\d+)(?:st|nd|rd|th)?\s+(?:policy\s+)?(?:year|yr)s?|(?:year|yr)s?\s+(\d+)",
    re.IGNORECASE,
)
_RENEWAL_DIGIT_RE = re.compile(r"(\d+)(?:st|nd|rd|th)?\s+renewal", re.IGNORECASE)
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
}
_YEAR_OR_RENEWAL_WORD_RE = re.compile(
    r"\b(" + "|".join(_ORDINAL_WORDS) + r")\s+(?:policy\s+)?(year|renewal)s?\b",
    re.IGNORECASE,
)

_AGE_RE = re.compile(
    r"(\d+)\s+(?:year(?:s)?(?:\s+old)?|yr(?:s)?(?:\s+old)?)\s+(?:old\s+)?(?:vehicle|bike|scooter|tw|two[\s\-]?wheeler)",
    re.IGNORECASE,
)
_CLAIMS_RE = re.compile(
    r"(\d+)\s+claim", re.IGNORECASE
)


def _parse_amount(num_str: str, suffix: str | None) -> Optional[float]:
    try:
        val = float(num_str.replace(",", ""))
    except (ValueError, AttributeError):
        return None
    if suffix:
        val *= _AMOUNT_MULTIPLIERS.get(suffix.lower(), 1)
    return val


def _extract_premium(query: str) -> Optional[float]:
    """Extract a monetary amount, trying 'premium 5000' then '5000 rs' patterns."""
    m = _PREMIUM_RE.search(query)
    if m:
        return _parse_amount(m.group(1), m.group(2))
    m = _AMOUNT_FALLBACK_RE.search(query)
    if m:
        num = m.group(1) or m.group(3)
        suf = m.group(2) or m.group(4)
        return _parse_amount(num, suf)
    return None


def _extract_price(query: str) -> Optional[float]:
    """Extract a price/cost amount from the query."""
    m = _PRICE_RE.search(query)
    if m:
        return _parse_amount(m.group(1), m.group(2))
    m = _AMOUNT_FALLBACK_RE.search(query)
    if m:
        num = m.group(1) or m.group(3)
        suf = m.group(2) or m.group(4)
        return _parse_amount(num, suf)
    return None


def _extract_policy_year(query: str) -> Optional[int]:
    """Resolve a policy year from digit years/renewals or spelled-out ordinals."""
    m = _YEAR_RE.search(query)
    if m:
        raw = m.group(1) or m.group(2)
        return int(raw) if raw else None

    m = _RENEWAL_DIGIT_RE.search(query)
    if m:
        return int(m.group(1)) + 1

    m = _YEAR_OR_RENEWAL_WORD_RE.search(query)
    if m:
        n = _ORDINAL_WORDS[m.group(1).lower()]
        return n + 1 if m.group(2).lower() == "renewal" else n

    return None


def detect_and_calculate(query: str) -> Optional[dict]:
    """Returns None if the query isn't a calculation request, so the caller
    falls back to normal RAG retrieval."""

    if _NCB_INTENT_RE.search(query):
        premium = _extract_premium(query)
        year = _extract_policy_year(query)
        claims_match = _CLAIMS_RE.search(query)

        if premium and year:
            claims = int(claims_match.group(1)) if claims_match else 0
            result = calculate_ncb(premium, year, claims)
            return {
                "calc_type": "ncb",
                "result": result,
                "formatted_answer": _format_ncb_answer(result),
            }

    if _IDV_INTENT_RE.search(query):
        price = _extract_price(query)
        age_match = _AGE_RE.search(query)

        if price and age_match:
            age = int(age_match.group(1))
            result = calculate_idv(price, age)
            return {
                "calc_type": "idv",
                "result": result,
                "formatted_answer": _format_idv_answer(result),
            }

    return None


def _format_ncb_answer(r: dict) -> str:
    lines = [
        "## NCB Calculation",
        "",
        r["step_by_step"],
        "",
        f"**NCB Discount: Rs.{r['discount_rs']}**",
        f"**Payable OD Premium: Rs.{r['payable_od_premium_rs']}**",
        "",
        f"NOTE: {r['note']}",
    ]
    return "\n".join(lines)


def _format_idv_answer(r: dict) -> str:
    lines = [
        "## IDV Calculation",
        "",
        r["step_by_step"],
        "",
    ]
    if isinstance(r.get("idv_rs"), (int, float)):
        lines.append(f"**IDV (Sum Insured): Rs.{r['idv_rs']}**")
    else:
        lines.append(f"**IDV: {r.get('idv_rs')}**")
    lines += ["", f"NOTE: {r['note']}"]
    return "\n".join(lines)
