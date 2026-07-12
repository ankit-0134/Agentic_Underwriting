import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import pdfplumber
import fitz
import base64
import io
import json
import asyncio
from openai import OpenAI

from rag_classify import build_dashboard_payload

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ─────────────────────────────────────
# HELPER — Detect digital or scanned
# ─────────────────────────────────────
def is_digital_pdf(file_bytes: bytes) -> bool:
    doc  = fitz.open(stream=file_bytes, filetype="pdf")
    text = doc[0].get_text().strip()
    return len(text) > 50


# ─────────────────────────────────────
# HELPER — Extract text from digital PDF
# ─────────────────────────────────────
def extract_digital_pdf(file_bytes: bytes) -> str:
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text.strip()


# ─────────────────────────────────────
# HELPER — Extract from scanned PDF or image
# ─────────────────────────────────────
def extract_with_vision(file_bytes: bytes, filename: str) -> str:
    from pdf2image import convert_from_bytes
    from PIL import Image

    if filename.lower().endswith(".pdf"):
        images    = convert_from_bytes(
            file_bytes,
            dpi=200,
            poppler_path=r"C:\poppler\poppler-26.02.0\Library\bin"
        )
        pil_image = images[0]
    else:
        pil_image = Image.open(io.BytesIO(file_bytes))

    buffer       = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_base64}"}
                },
                {
                    "type": "text",
                    "text": "Extract all text from this document. Return the raw text only — no formatting, no explanation."
                }
            ]
        }],
        temperature=0,
        max_tokens=2000
    )

    return response.choices[0].message.content


# ─────────────────────────────────────
# HELPER — Auto detect and extract
# ─────────────────────────────────────
def auto_extract(file_bytes: bytes, filename: str) -> dict:
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        method = "gpt4o_vision"
        text   = extract_with_vision(file_bytes, filename)

    elif filename.lower().endswith(".pdf"):
        if is_digital_pdf(file_bytes):
            method = "pdfplumber"
            text   = extract_digital_pdf(file_bytes)
        else:
            method = "gpt4o_vision"
            text   = extract_with_vision(file_bytes, filename)
    else:
        method = "unknown"
        text   = ""

    return {
        "filename":          filename,
        "extraction_method": method,
        "extracted_text":    text
    }


# ─────────────────────────────────────
# ENDPOINT — Extract single file
# ─────────────────────────────────────
@app.post("/extract")
async def extract_file(file: UploadFile = File(...)):
    file_bytes = await file.read()
    result     = auto_extract(file_bytes, file.filename)
    return result


# ─────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────
app_prompt = """
You are an insurance application document expert.
Extract all fields from these application documents.

CRITICAL RULES FOR EXTRACTION:

RULE 1 — BOOLEAN FIELDS MUST BE TRUE JSON BOOLEANS:
  Any field answering a Yes/No question (e.g. works_underground, smoker,
  diabetes, hypertension, existing_insurance, scuba_diving, parachuting,
  licence_suspended_ever, positive_drug_screen, etc.) MUST be output as the
  JSON boolean true or false.
  NEVER output the strings "YES", "NO", "Yes", "No" for these fields —
  convert them: "YES" → true, "NO" → false.

RULE 2 — NUMERIC FIELDS MUST BE PLAIN NUMBERS:
  Any field that is a count, amount, age, year, or measurement (sum_assured,
  policy_term_years, number_of_dependents, age, scuba_max_depth_metres,
  scuba_annual_dives, mountaineering_max_altitude_metres,
  private_aviation_hours_per_year, parachuting_jumps_per_year,
  hang_gliding_flights_per_year, tobacco_quit_months,
  alcohol_frequency_per_week, DUI_convictions_last_12_months,
  accidents_last_12_months, moving_violations_last_3_years,
  existing_coverage_amount, total_coverage_applied, cibil_score, etc.)
  MUST be a plain number — strip currency symbols, commas, units, and
  words like "Years", "metres", "dives annually", "times per week".
  Example: "Rs. 2,00,00,000 (Two Crore Only)" → 20000000
  Example: "1 time per week" → 1
  Example: "38 Years" → 38
  Fields explicitly named "*_display" keep the original human-readable text
  instead (e.g. sum_assured_display: "Rs. 2,00,00,000").

RULE 3 — APPLICATION / PROPOSAL NUMBER:
  "application_number" is usually printed as "Proposal No" near the top
  of the document header, even if it appears outside the main personal
  details section — extract it from wherever it appears in the document.

Return ONLY a valid JSON object with EXACTLY this structure.
Use null for any field not found. No explanation. JSON only.

{
  "personal": {
    "application_number": null,
    "full_name": null,
    "date_of_birth": null,
    "age": null,
    "gender": null,
    "marital_status": null,
    "number_of_dependents": null,
    "address": null,
    "city": null,
    "state": null,
    "pincode": null,
    "nationality": null
  },
  "occupation": {
    "designation": null,
    "employer": null,
    "nature_of_work": null,
    "works_underground": null,
    "works_offshore": null,
    "works_at_height": null,
    "works_with_explosives": null,
    "armed_occupation": null
  },
  "avocations": {
    "scuba_diving": null,
    "scuba_max_depth_metres": null,
    "scuba_annual_dives": null,
    "scuba_cave_diving": null,
    "scuba_formally_trained": null,
    "mountaineering": null,
    "mountaineering_max_altitude_metres": null,
    "private_aviation": null,
    "private_aviation_role": null,
    "private_aviation_hours_per_year": null,
    "parachuting": null,
    "parachuting_type": null,
    "parachuting_jumps_per_year": null,
    "hang_gliding": null,
    "hang_gliding_type": null,
    "hang_gliding_flights_per_year": null,
    "hot_air_ballooning": null,
    "hot_air_ballooning_role": null,
    "motorsports": null,
    "motorsports_type": null
  },
  "lifestyle": {
    "smoker": null,
    "tobacco_type": null,
    "tobacco_quit_months": null,
    "alcohol_use": null,
    "alcohol_frequency_per_week": null,
    "marijuana_use": null,
    "cannabis_frequency": null,
    "positive_drug_screen": null
  },
  "credit": {
    "cibil_score": null
  },
  "health_declaration": {
    "diabetes": null,
    "diabetes_type": null,
    "diabetes_since_year": null,
    "hypertension": null,
    "hypertension_since_year": null,
    "heart_disease": null,
    "kidney_disease": null,
    "liver_disease": null,
    "cancer": null,
    "any_surgery_last_5_years": null,
    "any_hospitalization_last_3_years": null,
    "current_medications": []
  },
  "family_history": {
    "father_heart_disease": null,
    "father_age_at_death": null,
    "mother_diabetes": null,
    "mother_alive": null,
    "siblings_heart_disease": null,
    "siblings_cancer": null
  },
  "driving_record": {
    "DUI_convictions_last_12_months": null,
    "reckless_driving_convictions": null,
    "moving_violations_last_3_years": null,
    "licence_suspended_ever": null,
    "accidents_last_12_months": null
  },
  "policy": {
    "sum_assured": null,
    "sum_assured_display": null,
    "policy_term_years": null,
    "existing_insurance": null,
    "existing_coverage_amount": null,
    "total_coverage_applied": null
  }
}
"""

fin_prompt = """
You are a financial document expert.
Extract all fields from these financial documents
(ITR, salary slip, bank statement).
Application context is also provided for sum assured.

CRITICAL RULES FOR EXTRACTION:

RULE 1 — ALWAYS include currency:
  "Rs. 9,01,200" not "901200"

RULE 2 — CROSS DOCUMENT VERIFICATION:
  Income will appear in multiple documents:
  - ITR         → annual income (government verified)
  - Salary Slip → monthly gross and net
  - Bank Stmt   → actual monthly credit received
  Compare all three and note any mismatch.
  Use ITR annual income as primary source.

RULE 3 — CALCULATE THESE YOURSELF:

  annual_income:
    If only monthly_gross_salary is available:
    annual_income = monthly_gross_salary x 12

  monthly_gross_salary:
    If only annual_income is available:
    monthly_gross_salary = annual_income / 12

  total_annual_income:
    = annual_income + variable_pay_annual
    If variable_pay not found: total_annual_income = annual_income

  income_coverage_ratio:
    = sum_assured / annual_income
    Round to 1 decimal. Store as: "22.2x"
    sum_assured comes from Application Context above

  age_band:
    Find applicant age from documents.
    Assign band:
    Age 20-40  → "20-40"
    Age 41-50  → "41-50"
    Age 51-55  → "51-55"
    Age 56-65  → "56-65"
    Age 66+    → "66+"

  recommended_max_ratio:
    Based on age_band:
    "20-40"  → "25x"
    "41-50"  → "20x"
    "51-55"  → "15x"
    "56-65"  → "10x"
    "66+"    → "7x"

  ratio_within_limit:
    Compare income_coverage_ratio with recommended_max_ratio
    true  if ratio <= recommended limit
    false if ratio exceeds recommended limit

  EMI_to_income_ratio:
    = total_monthly_EMI / monthly_net_salary x 100
    Round to 1 decimal. Store as: "52.9%"
    Identify EMIs from bank statement:
    Look for recurring debits labelled EMI, LOAN, ECS

  income_trend:
    If ITR has 2 years of data:
      year2 > year1 → "Increasing"
      year2 = year1 → "Stable"
      year2 < year1 → "Decreasing"
    If only 1 year available → "Stable"

  months_salary_missing:
    Count months in bank statement where
    no salary credit was received.
    Store as number: 0, 1, 2 etc.

  average_monthly_balance:
    Calculate from closing balances across months
    Store as: "Rs. 42,000"

RULE 4 — LARGE UNUSUAL TRANSACTIONS:
  Flag any single transaction above Rs. 1,00,000
  that is NOT a regular salary or EMI.
  Store as list of objects:
  [{"date": "15-Mar-2024", "amount": "Rs. 5,00,000",
    "description": "NEFT CR UNKNOWN SOURCE"}]

RULE 5 — INCOME CONSISTENCY CHECK:
  Compare:
    ITR annual income
    Salary slip annual income (monthly x 12)
    Bank statement average monthly credit x 12
  If any two differ by more than 20% — flag it:
  income_inconsistency: true

RULE 6 — DEBIT OR CREDIT IDENTIFICATION IN BANK STATEMENT:
  Column positions are unreliable in extracted text.
  Use these two methods:

  METHOD 1 — Description keywords:
    CR, CREDIT, SALARY, INTEREST        → Credit
    DR, DEBIT, EMI, ECS                 → Debit
    ATM WDL, ATM WITHDRAWAL             → Debit
    UPI (most cases)                    → Debit
    NEFT CR                             → Credit
    NEFT DR                             → Debit

  METHOD 2 — Balance change (most reliable):
    Current balance > Previous balance  → Credit
    Credit amount = Current - Previous
    Current balance < Previous balance  → Debit
    Debit amount = Previous - Current

  LAST COLUMN IS ALWAYS RUNNING BALANCE:
    Never treat last column as debit or credit
    Use it only to determine transaction direction

  For EMI detection:
    Look for EMI, ECS, LOAN in description
    Same amount repeating every month
    Always Debit

RULE 7 — SALARY VERIFICATION FROM EMPLOYER NAME:
  Step 1 — Get employer name from salary slip
  Step 2 — Search bank statement for credit transactions
           containing employer name keywords
  Step 3 — Check if salary credit appears every month
  Step 4 — Verify amount matches salary slip net salary

RULE 8 — TOTAL DEBIT AND CREDIT CALCULATION:
  Sum ALL credit transactions in bank statement
  Sum ALL debit transactions in bank statement
  Opening balance + Total credits - Total debits = Closing balance

RULE 9 — NAME AND AGE PER SOURCE DOCUMENT:
  "income.name"       → full name exactly as printed on the salary slip
  "itr.name"           → full name exactly as printed on the ITR
  "itr.date_of_birth"  → date of birth exactly as printed on the ITR / PAN
                         details section, if shown. ITRs almost always show
                         date of birth, not age — extract it here verbatim.
  "itr.age"            → applicant age as of 31st March of the assessment
                         year shown on the ITR. Calculate it yourself:
                         age = (assessment year's ending calendar year) - (birth year),
                         then subtract 1 if the birthday falls after 31st March.
                         Example: DOB 15-Aug-1988, Assessment Year 2024-25
                         (ends 31-Mar-2025) → 2025 - 1988 = 37, birthday
                         (15-Aug) is after 31-Mar, so final age = 36.
  "bank_summary.name"  → full name / account holder name exactly as printed
                         on the bank statement
  Do NOT copy the application form's name into these fields — each must
  reflect only what that specific document itself states, even if it
  differs from the other documents. This is used later for cross-document
  name and age conflict checks.

RULE 10 — CIBIL / CREDIT SCORE:
  Only fill "credit.cibil_score" if a credit bureau report (CIBIL, Experian,
  CRIF, etc.) is among the uploaded financial documents.
  If no such report was provided, leave "credit.cibil_score" as null —
  do NOT guess or estimate it from income or EMI data.

Return ONLY a valid JSON object with EXACTLY this structure.
Use null for any field not found. No explanation. JSON only.

{
  "income": {
    "name": null,
    "annual_income": null,
    "annual_income_display": null,
    "income_source": null,
    "employer": null,
    "monthly_gross_salary": null,
    "monthly_net_salary": null,
    "variable_pay_annual": null,
    "total_annual_income": null,
    "income_trend": null,
    "income_inconsistency": null,
    "itr_annual_income": null,
    "salary_slip_annual_income": null,
    "bank_annual_income": null
  },
  "itr": {
    "name": null,
    "date_of_birth": null,
    "age": null,
    "assessment_year": null,
    "gross_total_income": null,
    "net_taxable_income": null,
    "tax_paid": null,
    "pan_verified": null,
    "itr_filed": null
  },
  "coverage_analysis": {
    "sum_assured": null,
    "annual_income": null,
    "income_coverage_ratio": null,
    "age_band": null,
    "recommended_max_ratio": null,
    "ratio_within_limit": null
  },
  "bank_summary": {
    "name": null,
    "opening_balance": null,
    "closing_balance": null,
    "total_credits": null,
    "total_debits": null,
    "balance_verified": null,
    "average_monthly_balance": null,
    "salary_credits_regular": null,
    "salary_credit_amount": null,
    "salary_verified": null,
    "salary_mismatch": null,
    "employer_from_slip": null,
    "salary_keyword_in_bank": null,
    "months_salary_missing": null,
    "total_monthly_EMI": null,
    "EMI_to_income_ratio": null,
    "large_unusual_credits": [],
    "large_unusual_debits": []
  },
  "credit": {
    "cibil_score": null
  }
}
"""

med_prompt = """
You are a medical document expert.
Extract all fields from these medical documents.

CRITICAL RULES FOR UNITS:
1. ALWAYS include the unit for every numeric medical value
2. If unit is clearly mentioned in document — use that exact unit
3. If unit is NOT mentioned — infer the standard clinical unit:
   - HbA1c          → %
   - Blood glucose   → mg/dL
   - Cholesterol     → mg/dL
   - Triglycerides   → mg/dL
   - Creatinine      → mg/dL
   - TSH             → uIU/mL
   - Hemoglobin      → g/dL
   - Blood pressure  → mmHg
   - Height          → cm
   - Weight          → kg
   - BMI             → kg/m2
   - Pulse           → bpm
   - SpO2            → %
   - eGFR            → ml/min/1.73m2
   - Bilirubin       → mg/dL
   - SGOT/SGPT       → U/L
   - Sodium/Potassium→ mEq/L
4. Store value and unit together as string: "7.8 %" or "142 mg/dL"
5. NEVER store a numeric value without its unit

Return ONLY a valid JSON object with EXACTLY this structure.
Use null for any field not found. No explanation. JSON only.

{
  "physical": {
    "height": null,
    "weight": null,
    "BMI": null,
    "waist": null,
    "blood_pressure": null,
    "BP_systolic": null,
    "BP_diastolic": null,
    "pulse_rate": null,
    "SpO2": null
  },
  "blood_glucose": {
    "fasting_blood_sugar": null,
    "post_prandial_glucose": null,
    "HbA1c": null,
    "estimated_avg_glucose": null
  },
  "lipid_profile": {
    "total_cholesterol": null,
    "HDL_cholesterol": null,
    "LDL_cholesterol": null,
    "VLDL_cholesterol": null,
    "triglycerides": null,
    "cholesterol_ratio": null,
    "LDL_HDL_ratio": null
  },
  "liver_function": {
    "total_bilirubin": null,
    "AST_SGOT": null,
    "ALT_SGPT": null,
    "alkaline_phosphatase": null,
    "total_protein": null,
    "albumin": null,
    "GGT": null
  },
  "kidney_function": {
    "blood_urea": null,
    "serum_creatinine": null,
    "uric_acid": null,
    "eGFR": null,
    "sodium": null,
    "potassium": null
  },
  "thyroid": {
    "TSH": null,
    "T3": null,
    "T4": null
  },
  "cbc": {
    "hemoglobin": null,
    "WBC_count": null,
    "RBC_count": null,
    "platelet_count": null,
    "hematocrit": null
  },
  "infectious": {
    "HIV": null,
    "HBsAg": null,
    "anti_HCV": null,
    "VDRL": null
  },
  "urine": {
    "sugar": null,
    "protein": null,
    "ketones": null
  },
  "ecg": {
    "ECG_result": null,
    "heart_rate_bpm": null,
    "rhythm": null,
    "ST_changes": null,
    "LVH": null,
    "bundle_branch_block": null,
    "atrial_fibrillation": null,
    "overall_interpretation": null
  },
  "history": {
    "diabetes": null,
    "diabetes_type": null,
    "hypertension": null,
    "heart_disease": null,
    "kidney_disease": null,
    "cancer": null,
    "current_medications": []
  }
}
"""

# ─────────────────────────────────────
# HELPER — LLM extract
# ─────────────────────────────────────
async def llm_extract(text: str, category: str, prompt: str) -> dict:
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user",   "content": text}
        ],
        temperature=0,
        max_tokens=4096
    )
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": raw}


# ─────────────────────────────────────
# ENDPOINT — Analyse all 3 categories
# ─────────────────────────────────────
@app.post("/analyse")
async def analyse_case(
    application_files: List[UploadFile] = File(default=[]),
    financial_files:   List[UploadFile] = File(default=[]),
    medical_files:     List[UploadFile] = File(default=[])
):
    # Step 1 — Extract text from all files
    async def extract_category(files):
        combined_text = ""
        for f in files:
            file_bytes     = await f.read()
            result         = auto_extract(file_bytes, f.filename)
            combined_text += f"\n\nDocument: {f.filename}\n{result['extracted_text']}"
        return combined_text

    app_text, fin_text, med_text = await asyncio.gather(
        extract_category(application_files),
        extract_category(financial_files),
        extract_category(medical_files)
    )

    # Step 2 — 3 parallel LLM calls
    app_json, fin_json, med_json = await asyncio.gather(
        llm_extract(app_text, "application", app_prompt),
        llm_extract(
            f"APPLICATION CONTEXT (for sum assured and age):\n{app_text}\n\nFINANCIAL DOCUMENTS:\n{fin_text}",
            "financial",
            fin_prompt
        ),
        llm_extract(med_text, "medical", med_prompt)
    )

    # Step 3 — RAG-grounded risk classification + dashboard profile
    dashboard_payload = await build_dashboard_payload(app_json, fin_json, med_json)

    return {
        "application": app_json,
        "financial":   fin_json,
        "medical":     med_json,
        **dashboard_payload
    }


# ─────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────
@app.get("/")
def root():
    return {"status": "Underwriting Backend Running"}