"""
Queries the Chroma collections built by rag_setup.py for the specific
features shown on the frontend Dashboard, classifies each LOW/MEDIUM/HIGH
with a reason grounded in the retrieved rule + the extracted value, and
computes the 3 information-mismatch checks (name/age/income).
"""

import os
import json
import re
import asyncio
from openai import OpenAI

from rag_setup import OPENAI_API_KEY, CHROMA_DIR

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

client = OpenAI(api_key=OPENAI_API_KEY)
_embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

_collections = {}


def _get_collection(name):
    if name not in _collections:
        _collections[name] = Chroma(
            collection_name=name,
            embedding_function=_embeddings,
            persist_directory=CHROMA_DIR
        )
    return _collections[name]


def _fmt(value, default="Not available"):
    if value is None or value == "":
        return default
    return str(value)


def _parse_number(value):
    """Extracted fields sometimes come back as '38 Years' or 'Rs. 9,01,200'
    instead of a plain number — pull the first numeric token out."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group()) if match else None


def _resolve_total_monthly_emi(bank_summary, monthly_net_salary):
    """The LLM sometimes derives EMI_to_income_ratio from recurring debits
    without also filling total_monthly_EMI (e.g. when no transaction is
    cleanly labelled EMI/LOAN/ECS) — back-calculate it from the ratio and
    net salary rather than showing a blank."""
    parsed_emi = _parse_number(bank_summary.get("total_monthly_EMI"))
    if parsed_emi is not None:
        return parsed_emi
    ratio = _parse_number(bank_summary.get("EMI_to_income_ratio"))
    net_salary = _parse_number(monthly_net_salary)
    if ratio is None or net_salary is None:
        return None
    return round(ratio / 100 * net_salary)


def _format_inr(amount):
    """Indian digit grouping, e.g. 1200000 -> 'Rs. 12,00,000' — mirrors the
    format the extraction prompts use for *_display fields."""
    if amount is None:
        return None
    amount = int(round(amount))
    sign = "-" if amount < 0 else ""
    digits = str(abs(amount))
    if len(digits) <= 3:
        grouped = digits
    else:
        last3, rest = digits[-3:], digits[:-3]
        parts = []
        while len(rest) > 2:
            parts.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            parts.insert(0, rest)
        grouped = ",".join(parts) + "," + last3
    return f"{sign}Rs. {grouped}"


def _resolve_annual_income(income, coverage, policy):
    """annual_income is carried redundantly across income.annual_income,
    income.itr/salary_slip/bank component figures, and coverage_analysis —
    or it can be reverse-derived from sum_assured and the coverage ratio the
    LLM already computed. Same class of drop as total_monthly_EMI: the model
    fills the derived ratio field but skips restating the source figure."""
    for source in (
        income.get("annual_income"),
        income.get("itr_annual_income"),
        income.get("salary_slip_annual_income"),
        income.get("bank_annual_income"),
        coverage.get("annual_income"),
    ):
        parsed = _parse_number(source)
        if parsed is not None:
            return parsed

    ratio = _parse_number(coverage.get("income_coverage_ratio"))
    sum_assured = _parse_number(policy.get("sum_assured") or coverage.get("sum_assured"))
    if ratio and sum_assured:
        return round(sum_assured / ratio)
    return None


def _resolve_monthly_net_salary(income, resolved_annual_income):
    direct = _parse_number(income.get("monthly_net_salary"))
    if direct is not None:
        return direct
    gross = _parse_number(income.get("monthly_gross_salary"))
    if gross is not None:
        return gross
    annual = _parse_number(resolved_annual_income)
    if annual is not None:
        return round(annual / 12)
    return None


# ─────────────────────────────────────
# AGE BAND — mirrors the bands in rag_setup.py's income_replacement chunks
# ─────────────────────────────────────
def age_band_for(age):
    age = _parse_number(age)
    if age is None:
        return None
    age = int(age)
    if age <= 40:
        return "20-40"
    if age <= 50:
        return "41-50"
    if age <= 55:
        return "51-55"
    if age <= 65:
        return "56-65"
    return "66+"


# ─────────────────────────────────────
# RULE RETRIEVAL — real RAG: embed the query, similarity-search the collection
# ─────────────────────────────────────
def retrieve_rule(collection_name, query, k=2, exclude_topics=None):
    """Embeds `query` and runs a nearest-neighbour similarity search against
    the named Chroma collection — this is the actual RAG call. `query` should
    be a natural-language description that includes the applicant's real
    extracted value, so retrieval is grounded in their specific case."""
    if not query:
        return None
    collection = _get_collection(collection_name)
    results = collection.similarity_search(query, k=k)
    docs = [
        doc.page_content for doc in results
        if not exclude_topics or doc.metadata.get("topic") not in exclude_topics
    ]
    return "\n\n".join(docs) if docs else None


_OCCUPATION_EXCLUDE_TOPICS = {
    "smoking_nicotine", "drunk_driving",
    "avocation_scuba", "avocation_mountaineering", "avocation_racing",
    "avocation_aviation_airborne"
}


# ─────────────────────────────────────
# FEATURE BUILDERS — one function per dashboard section
# ─────────────────────────────────────
def build_financial_features(app_json, fin_json):
    app_json = app_json or {}
    fin_json = fin_json or {}
    personal = app_json.get("personal", {}) or {}
    coverage = fin_json.get("coverage_analysis", {}) or {}
    bank = fin_json.get("bank_summary", {}) or {}
    credit = fin_json.get("credit", {}) or {}
    app_credit = app_json.get("credit", {}) or {}

    age = personal.get("age")
    band = age_band_for(age)
    ratio = coverage.get("income_coverage_ratio")
    emi_ratio = bank.get("EMI_to_income_ratio")
    cibil = credit.get("cibil_score") or app_credit.get("cibil_score")

    return [
        {
            "feature": "Income Coverage Ratio",
            "value": f"{_fmt(ratio)} (age band {band or 'unknown'}, applicant age {_fmt(age)})",
            "rule": retrieve_rule(
                "collection_financial",
                f"income replacement maximum coverage multiple for applicant age {_fmt(age)} "
                f"in age band {band or 'unknown'}, income coverage ratio {_fmt(ratio)}"
            )
        },
        {
            "feature": "EMI to Income Ratio",
            "value": _fmt(emi_ratio),
            "rule": retrieve_rule(
                "collection_financial",
                f"EMI to income ratio {_fmt(emi_ratio)} financial stress assessment"
            )
        },
        {
            "feature": "CIBIL Score",
            "value": _fmt(cibil, "Not available — no credit report uploaded"),
            "rule": retrieve_rule(
                "collection_financial",
                f"CIBIL credit score {_fmt(cibil)} credit risk assessment"
            )
        }
    ]


def build_medical_features(app_json, med_json):
    app_json = app_json or {}
    med_json = med_json or {}
    physical = med_json.get("physical", {}) or {}
    lipid = med_json.get("lipid_profile", {}) or {}
    glucose = med_json.get("blood_glucose", {}) or {}
    history = med_json.get("history", {}) or {}
    kidney = med_json.get("kidney_function", {}) or {}
    liver = med_json.get("liver_function", {}) or {}
    ecg = med_json.get("ecg", {}) or {}
    thyroid = med_json.get("thyroid", {}) or {}
    family = app_json.get("family_history", {}) or {}
    app_health = app_json.get("health_declaration", {}) or {}

    bp = physical.get("blood_pressure")
    if not bp and (physical.get("BP_systolic") or physical.get("BP_diastolic")):
        bp = f"{_fmt(physical.get('BP_systolic'), '?')}/{_fmt(physical.get('BP_diastolic'), '?')} mmHg"

    if app_health.get("hypertension"):
        medications = ", ".join(app_health.get("current_medications") or [])
        bp_value = (
            f"{_fmt(bp)} — diagnosed hypertension since {_fmt(app_health.get('hypertension_since_year'))}, "
            f"on medication: {_fmt(medications, 'not specified')}"
        )
    else:
        bp_value = _fmt(bp)

    diabetes_value = (
        f"{_fmt(history.get('diabetes_type'), 'Diabetic')} — FBS {_fmt(glucose.get('fasting_blood_sugar'))}, HbA1c {_fmt(glucose.get('HbA1c'))}"
        if history.get("diabetes") else "No diabetes reported"
    )

    family_value = (
        f"Father heart disease: {_fmt(family.get('father_heart_disease'), 'No')}"
        f" (age at death {_fmt(family.get('father_age_at_death'))}); "
        f"Sibling heart disease: {_fmt(family.get('siblings_heart_disease'), 'No')}"
    )

    chol_total = lipid.get("total_cholesterol")
    chol_ratio = lipid.get("cholesterol_ratio")
    bmi = physical.get("BMI")
    creatinine = kidney.get("serum_creatinine")
    egfr = kidney.get("eGFR")
    ast = liver.get("AST_SGOT")
    alt = liver.get("ALT_SGPT")
    bilirubin = liver.get("total_bilirubin")
    ecg_interp = ecg.get("overall_interpretation")
    hr = ecg.get("heart_rate_bpm")
    tsh = thyroid.get("TSH")

    return [
        {"feature": "Blood Pressure", "value": bp_value,
         "rule": retrieve_rule("collection_medical", f"blood pressure {bp_value} risk classification", k=3)},
        {"feature": "Cholesterol & Lipid Profile",
         "value": f"Total {_fmt(chol_total)}, ratio {_fmt(chol_ratio)}",
         "rule": retrieve_rule("collection_medical", f"cholesterol {_fmt(chol_total)} lipid profile ratio {_fmt(chol_ratio)} risk classification")},
        {"feature": "BMI", "value": _fmt(bmi),
         "rule": retrieve_rule("collection_medical", f"BMI {_fmt(bmi)} build risk classification")},
        {"feature": "Diabetes", "value": diabetes_value,
         "rule": retrieve_rule("collection_medical", f"diabetes {diabetes_value} blood sugar risk classification")},
        {"feature": "Family History — CAD", "value": family_value,
         "rule": retrieve_rule("collection_medical", f"family history coronary artery disease {family_value} risk classification", k=3)},
        {"feature": "Kidney Function",
         "value": f"Creatinine {_fmt(creatinine)}, eGFR {_fmt(egfr)}",
         "rule": retrieve_rule("collection_medical", f"kidney function creatinine {_fmt(creatinine)} eGFR {_fmt(egfr)} risk classification")},
        {"feature": "Liver Function",
         "value": f"AST {_fmt(ast)}, ALT {_fmt(alt)}, Bilirubin {_fmt(bilirubin)}",
         "rule": retrieve_rule("collection_medical", f"liver function AST {_fmt(ast)} ALT {_fmt(alt)} bilirubin {_fmt(bilirubin)} risk classification")},
        {"feature": "ECG",
         "value": f"{_fmt(ecg_interp)}, HR {_fmt(hr)}",
         "rule": retrieve_rule("collection_medical", f"ECG {_fmt(ecg_interp)} heart rate {_fmt(hr)} bpm risk classification")},
        {"feature": "Thyroid Function", "value": f"TSH {_fmt(tsh)}",
         "rule": retrieve_rule("collection_medical", f"thyroid TSH {_fmt(tsh)} risk classification")},
    ]


def _avocations_summary(avocations):
    avocations = avocations or {}
    parts = []
    if avocations.get("scuba_diving"):
        parts.append(
            f"Scuba diving (max depth {_fmt(avocations.get('scuba_max_depth_metres'))}, "
            f"{_fmt(avocations.get('scuba_annual_dives'))} dives/year, "
            f"cave diving: {_fmt(avocations.get('scuba_cave_diving'), 'No')}, "
            f"certified: {_fmt(avocations.get('scuba_formally_trained'), 'No')})"
        )
    if avocations.get("mountaineering"):
        parts.append(f"Mountaineering (max altitude {_fmt(avocations.get('mountaineering_max_altitude_metres'))})")
    if avocations.get("private_aviation"):
        parts.append(
            f"Private aviation ({_fmt(avocations.get('private_aviation_role'), 'role not specified')}, "
            f"{_fmt(avocations.get('private_aviation_hours_per_year'))} hrs/year)"
        )
    if avocations.get("parachuting"):
        parts.append(
            f"Parachuting ({_fmt(avocations.get('parachuting_type'), 'type not specified')}, "
            f"{_fmt(avocations.get('parachuting_jumps_per_year'))} jumps/year)"
        )
    if avocations.get("hang_gliding"):
        parts.append(
            f"Hang gliding ({_fmt(avocations.get('hang_gliding_type'), 'type not specified')}, "
            f"{_fmt(avocations.get('hang_gliding_flights_per_year'))} flights/year)"
        )
    if avocations.get("hot_air_ballooning"):
        parts.append(f"Hot air ballooning ({_fmt(avocations.get('hot_air_ballooning_role'), 'role not specified')})")
    if avocations.get("motorsports"):
        parts.append(f"Motorsports ({_fmt(avocations.get('motorsports_type'), 'type not specified')})")
    return "; ".join(parts) if parts else "No hazardous avocations or adventure activities declared"


def build_lifestyle_features(app_json):
    app_json = app_json or {}
    occupation = app_json.get("occupation", {}) or {}
    avocations = app_json.get("avocations", {}) or {}
    lifestyle = app_json.get("lifestyle", {}) or {}
    driving = app_json.get("driving_record", {}) or {}

    occ_name = occupation.get("designation")

    avocation_value = _avocations_summary(avocations)
    has_avocation = any(
        avocations.get(k)
        for k in ("scuba_diving", "mountaineering", "private_aviation",
                   "parachuting", "hang_gliding", "hot_air_ballooning", "motorsports")
    )

    smoker = lifestyle.get("smoker")
    quit_months = lifestyle.get("tobacco_quit_months")
    if smoker:
        smoke_value = f"Current tobacco/nicotine user (type: {_fmt(lifestyle.get('tobacco_type'))})"
    elif quit_months is not None:
        smoke_value = f"Former user, nicotine-free for {_fmt(quit_months)} months"
    else:
        smoke_value = "Non-smoker, nicotine-free"

    if lifestyle.get("marijuana_use"):
        smoke_value += f"; cannabis use ({_fmt(lifestyle.get('cannabis_frequency'), 'frequency not specified')})"
    if lifestyle.get("positive_drug_screen"):
        smoke_value += "; positive drug screen on record"

    alcohol_use = lifestyle.get("alcohol_use")
    alcohol_freq = lifestyle.get("alcohol_frequency_per_week")
    alcohol_value = f"{_fmt(alcohol_use)} — {_fmt(alcohol_freq)} times/week"

    dui = driving.get("DUI_convictions_last_12_months")
    dui_value = f"{_fmt(dui, '0')} incidents in last 12 months"

    accidents = driving.get("accidents_last_12_months")
    accidents_value = f"{_fmt(accidents, '0')} accident(s) in last 12 months"

    return [
        {"feature": "Occupation", "value": _fmt(occ_name),
         "rule": retrieve_rule(
             "collection_lifestyle", _fmt(occ_name, ""), k=3,
             exclude_topics=_OCCUPATION_EXCLUDE_TOPICS
         )},
        {"feature": "Avocations", "value": avocation_value,
         "rule": retrieve_rule(
             "collection_lifestyle", f"avocations adventure activities {avocation_value} risk classification", k=3
         ) if has_avocation else None},
        {"feature": "Smoking / Nicotine", "value": smoke_value,
         "rule": retrieve_rule("collection_lifestyle", f"smoking tobacco nicotine use {smoke_value} risk classification")},
        {"feature": "Alcohol Use", "value": alcohol_value,
         "rule": retrieve_rule("collection_lifestyle", f"alcohol use frequency {alcohol_value} risk classification")},
        {"feature": "Drunk Driving", "value": dui_value,
         "rule": retrieve_rule("collection_lifestyle", f"drunk driving {dui_value} DUI risk classification")},
        {"feature": "Accidents", "value": accidents_value,
         "rule": retrieve_rule("collection_lifestyle", f"accidents while driving {accidents_value} risk classification")},
    ]


# ─────────────────────────────────────
# LLM CLASSIFICATION — one call per category, grounded in retrieved rules
# ─────────────────────────────────────
async def classify_category(category_label, features):
    items_for_prompt = [
        {
            "feature": f["feature"],
            "extracted_value": f["value"],
            "underwriting_rule": f["rule"] or "No specific rule was retrieved from the knowledge base for this feature — use general clinical/financial judgement and say so in the reason."
        }
        for f in features
    ]

    system_prompt = f"""
You are an insurance underwriting risk classifier for the {category_label} category.

For each item you are given:
- extracted_value: the applicant's actual data extracted from their documents
- underwriting_rule: the specific underwriting rule retrieved from the knowledge base for that feature

Classify each feature's risk as exactly one of: LOW, MEDIUM, HIGH.
Base the classification STRICTLY on applying underwriting_rule to extracted_value.
If extracted_value indicates missing/unavailable data, classify as MEDIUM and say so in the reason.

Write "reason" as 1-2 sentences that reference the ACTUAL extracted value and the SPECIFIC
threshold or rule that was applied — never a generic explanation.

Return ONLY a JSON array, no markdown, no explanation, in this exact shape:
[
  {{"feature": "...", "value": "...", "risk": "LOW|MEDIUM|HIGH", "reason": "..."}}
]
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(items_for_prompt, indent=2)}
        ],
        temperature=0,
        max_tokens=3000
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    return [
        {"feature": f["feature"], "value": f["value"], "risk": "MEDIUM",
         "reason": "Automatic classification failed to parse — manual underwriter review required."}
        for f in features
    ]


# ─────────────────────────────────────
# INFORMATION MISMATCH — deterministic, no LLM
# ─────────────────────────────────────
def _normalize_name_tokens(name):
    if not name:
        return set()
    name = re.sub(r"[^a-zA-Z\s]", "", name).lower()
    return set(t for t in name.split() if t)


def compute_name_mismatch(fin_json):
    fin_json = fin_json or {}
    income_name = (fin_json.get("income") or {}).get("name")
    itr_name = (fin_json.get("itr") or {}).get("name")
    bank_name = (fin_json.get("bank_summary") or {}).get("name")

    value = f"Salary Slip: {income_name or 'N/A'} | ITR: {itr_name or 'N/A'} | Bank Statement: {bank_name or 'N/A'}"
    names = [n for n in [income_name, itr_name, bank_name] if n]

    if len(names) < 2:
        return {
            "feature": "Name Mismatch", "value": value, "risk": "MEDIUM",
            "reason": "Fewer than two of the three source documents (salary slip, ITR, bank statement) had an extractable name — unable to fully cross-verify."
        }

    token_sets = [_normalize_name_tokens(n) for n in names]
    overlaps = []
    for i in range(len(token_sets)):
        for j in range(i + 1, len(token_sets)):
            a, b = token_sets[i], token_sets[j]
            smaller = min(len(a), len(b)) or 1
            overlaps.append(len(a & b) / smaller)
    min_overlap = min(overlaps) if overlaps else 0

    if min_overlap >= 0.99:
        risk, note = "LOW", "all tokens match across sources (allowing for short-form/spelling variation)"
    elif min_overlap >= 0.5:
        risk, note = "MEDIUM", "some tokens match but not all — likely a short-form or partial name"
    else:
        risk, note = "HIGH", "little to no token overlap between the names"

    reason = f"Comparing names across salary slip, ITR, and bank statement ({', '.join(names)}): {note}."
    return {"feature": "Name Mismatch", "value": value, "risk": risk, "reason": reason}


def compute_age_mismatch(app_json, fin_json):
    app_age = (app_json or {}).get("personal", {}).get("age")
    itr_age = (fin_json or {}).get("itr", {}).get("age")

    value = f"Application: {_fmt(app_age)} yrs | ITR: {_fmt(itr_age)} yrs"

    app_age_num = _parse_number(app_age)
    itr_age_num = _parse_number(itr_age)

    if app_age_num is None or itr_age_num is None:
        return {
            "feature": "Age Mismatch", "value": value, "risk": "HIGH",
            "reason": "Age could not be extracted from one or both sources (application form, ITR) — treated as high risk pending manual verification."
        }

    diff = abs(int(app_age_num) - int(itr_age_num))
    if diff <= 1:
        risk = "LOW"
    elif diff <= 3:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    reason = f"Application states age {app_age}, ITR-derived age is {itr_age} — a difference of {diff} year(s)."
    return {"feature": "Age Mismatch", "value": value, "risk": risk, "reason": reason}


def compute_income_mismatch(fin_json):
    income = (fin_json or {}).get("income", {}) or {}
    itr_income = income.get("itr_annual_income")
    slip_income = income.get("salary_slip_annual_income")
    bank_income = income.get("bank_annual_income")

    value = f"ITR: {_fmt(itr_income)} | Salary Slip: {_fmt(slip_income)} | Bank (annualized): {_fmt(bank_income)}"
    values = [v for v in (_parse_number(x) for x in [itr_income, slip_income, bank_income]) if v is not None]

    if len(values) < 2:
        return {
            "feature": "Income Mismatch", "value": value, "risk": "MEDIUM",
            "reason": "Fewer than two income figures (ITR, salary slip, bank credits) were available for cross-verification."
        }

    max_v, min_v = max(values), min(values)
    pct_diff = ((max_v - min_v) / max_v * 100) if max_v else 0

    if pct_diff <= 10:
        risk = "LOW"
    elif pct_diff <= 20:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    within = "within" if pct_diff <= 20 else "exceeds"
    reason = f"Largest gap between reported income figures is {pct_diff:.1f}% (ITR vs salary slip vs bank credits) — {within} the 20% conflict threshold."
    return {"feature": "Income Mismatch", "value": value, "risk": risk, "reason": reason}


def compute_information_mismatch(app_json, fin_json):
    return [
        compute_name_mismatch(fin_json),
        compute_age_mismatch(app_json, fin_json),
        compute_income_mismatch(fin_json)
    ]


# ─────────────────────────────────────
# PROFILE — merges the 3 extracted JSONs into the Dashboard's expected shape
# ─────────────────────────────────────
def build_profile(app_json, fin_json, med_json):
    app_json = app_json or {}
    fin_json = fin_json or {}
    med_json = med_json or {}

    personal = app_json.get("personal", {}) or {}
    occupation = app_json.get("occupation", {}) or {}
    avocations = app_json.get("avocations", {}) or {}
    app_lifestyle = app_json.get("lifestyle", {}) or {}
    app_health = app_json.get("health_declaration", {}) or {}
    family_history = app_json.get("family_history", {}) or {}
    driving_record = app_json.get("driving_record", {}) or {}
    policy = app_json.get("policy", {}) or {}

    income = fin_json.get("income", {}) or {}
    coverage = fin_json.get("coverage_analysis", {}) or {}
    bank_summary = fin_json.get("bank_summary", {}) or {}
    credit = fin_json.get("credit", {}) or {}
    app_credit = app_json.get("credit", {}) or {}

    physical = med_json.get("physical", {}) or {}
    glucose = med_json.get("blood_glucose", {}) or {}
    lipid = med_json.get("lipid_profile", {}) or {}
    liver = med_json.get("liver_function", {}) or {}
    thyroid = med_json.get("thyroid", {}) or {}
    kidney = med_json.get("kidney_function", {}) or {}

    lifestyle = {
        **app_lifestyle,
        "drunk_driving_incidents_last_12_months": driving_record.get("DUI_convictions_last_12_months")
    }

    resolved_annual_income = _resolve_annual_income(income, coverage, policy)
    resolved_monthly_net_salary = _resolve_monthly_net_salary(income, resolved_annual_income)

    financial_summary = {
        "annual_income": resolved_annual_income,
        "annual_income_display": income.get("annual_income_display") or _format_inr(resolved_annual_income),
        "monthly_net_salary": resolved_monthly_net_salary,
        "total_monthly_emi": _resolve_total_monthly_emi(bank_summary, resolved_monthly_net_salary),
        "emi_to_income_ratio": bank_summary.get("EMI_to_income_ratio"),
        "sum_assured": policy.get("sum_assured") or coverage.get("sum_assured"),
        "sum_assured_display": policy.get("sum_assured_display"),
        "income_coverage_ratio": coverage.get("income_coverage_ratio"),
        "cibil_score": credit.get("cibil_score") or app_credit.get("cibil_score")
    }

    health_declaration = {
        **app_health,
        "blood_pressure": physical.get("blood_pressure"),
        "bmi": physical.get("BMI"),
        "spo2": physical.get("SpO2"),
        "pulse_rate": physical.get("pulse_rate"),
        "hbA1c": glucose.get("HbA1c"),
        "total_cholesterol": lipid.get("total_cholesterol"),
        "cholesterol_ratio": lipid.get("cholesterol_ratio"),
        "liver_ast": liver.get("AST_SGOT"),
        "liver_alt": liver.get("ALT_SGPT"),
        "liver_bilirubin": liver.get("total_bilirubin"),
        "thyroid_tsh": thyroid.get("TSH"),
        "kidney_creatinine": kidney.get("serum_creatinine"),
        "kidney_egfr": kidney.get("eGFR")
    }

    return {
        "personal": personal,
        "occupation": occupation,
        "avocations": avocations,
        "lifestyle": lifestyle,
        "financial_summary": financial_summary,
        "policy": policy,
        "health_declaration": health_declaration,
        "family_history": family_history,
        "driving_record": driving_record
    }


# ─────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────
async def build_dashboard_payload(app_json, fin_json, med_json):
    profile = build_profile(app_json, fin_json, med_json)

    financial_features = build_financial_features(app_json, fin_json)
    medical_features = build_medical_features(app_json, med_json)
    lifestyle_features = build_lifestyle_features(app_json)

    financial_risk, medical_risk, lifestyle_risk = await asyncio.gather(
        classify_category("Financial", financial_features),
        classify_category("Medical", medical_features),
        classify_category("Lifestyle", lifestyle_features)
    )

    risk_classification = {
        "financial": financial_risk,
        "medical": medical_risk,
        "lifestyle": lifestyle_risk,
        "information_mismatch": compute_information_mismatch(app_json, fin_json)
    }

    return {"profile": profile, "risk_classification": risk_classification}
