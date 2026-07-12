
import os
from pathlib import Path
import chromadb
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv(Path(__file__).parent / ".env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR     = os.getenv("CHROMA_DIR", r"C:\Users\ankit\Documents\Underwriting\backend\chroma_db")
# ─────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION 1 — FINANCIAL  (Page 16)
# ══════════════════════════════════════════════════════════════════════════════
def build_financial_chunks():
    return [

        # ── AGE BAND 20–40 ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Income Replacement — Age Band 20 to 40\n"
                "Applicable age range: 20 years to 40 years.\n"
                "Maximum coverage multiple: 25x annual earned income.\n"
                "Minimum annual income required: Rs. 3,00,000 per annum.\n"
                "\n"
                "Low risk: income_coverage_ratio <= 25x.\n"
                "Medium risk: income_coverage_ratio > 25x and <= 27.5x.\n"
                "High risk: income_coverage_ratio > 27.5x."
            ),
            metadata={"source": "financial", "topic": "income_replacement", "age_band": "20-40", "max_multiple": 25}
        ),

        # ── AGE BAND 41–50 ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Income Replacement — Age Band 41 to 50\n"
                "Applicable age range: 41 years to 50 years.\n"
                "Maximum coverage multiple: 20x annual earned income.\n"
                "Minimum annual income required: Rs. 3,00,000 per annum.\n"
                "\n"
                "Low risk: income_coverage_ratio <= 20x.\n"
                "Medium risk: income_coverage_ratio > 20x and <= 22x.\n"
                "High risk: income_coverage_ratio > 22x."
            ),
            metadata={"source": "financial", "topic": "income_replacement", "age_band": "41-50", "max_multiple": 20}
        ),

        # ── AGE BAND 51–55 ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Income Replacement — Age Band 51 to 55\n"
                "Applicable age range: 51 years to 55 years.\n"
                "Maximum coverage multiple: 15x annual earned income.\n"
                "Minimum annual income required: Rs. 3,00,000 per annum.\n"
                "\n"
                "Low risk: income_coverage_ratio <= 15x.\n"
                "Medium risk: income_coverage_ratio > 15x and <= 16.5x.\n"
                "High risk: income_coverage_ratio > 16.5x."
            ),
            metadata={"source": "financial", "topic": "income_replacement", "age_band": "51-55", "max_multiple": 15}
        ),

        # ── AGE BAND 56–65 ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Income Replacement — Age Band 56 to 65\n"
                "Applicable age range: 56 years to 65 years.\n"
                "Maximum coverage multiple: 10x annual earned income.\n"
                "Minimum annual income required: Rs. 3,00,000 per annum.\n"
                "\n"
                "Low risk: income_coverage_ratio <= 10x.\n"
                "Medium risk: income_coverage_ratio > 10x and <= 11x.\n"
                "High risk: income_coverage_ratio > 11x."
            ),
            metadata={"source": "financial", "topic": "income_replacement", "age_band": "56-65", "max_multiple": 10}
        ),

        # ── AGE BAND 66+ ─────────────────────────────────────────────────────
        Document(
            page_content=(
                "Income Replacement — Age 66 and Above\n"
                "Applicable age range: 66 years and above.\n"
                "Maximum coverage multiple: 7x annual earned income.\n"
                "Minimum annual income required: Rs. 3,00,000 per annum.\n"
                "\n"
                "Low risk: income_coverage_ratio <= 7x and applicant is actively employed.\n"
                "Medium risk: income_coverage_ratio > 7x and <= 7.7x.\n"
                "High risk: income_coverage_ratio > 7.7x or applicant is not actively employed."
            ),
            metadata={"source": "financial", "topic": "income_replacement", "age_band": "66+", "max_multiple": 7}
        ),

        # ── EMI TO INCOME RATIO ───────────────────────────────────────────────
        Document(
            page_content=(
                "EMI to Income Ratio — Financial Stress Assessment\n"
                "Formula: EMI_to_income_ratio = total_monthly_EMI / monthly_net_salary x 100\n"
                "\n"
                "Low risk: EMI ratio <= 40%.\n"
                "Medium risk: EMI ratio > 40% and <= 55%.\n"
                "High risk: EMI ratio > 55%."
            ),
            metadata={"source": "financial", "topic": "emi_to_income_ratio"}
        ),

        # ── CIBIL SCORE ──────────────────────────────────────────────────────
        Document(
            page_content=(
                "CIBIL Score — Credit Risk Assessment\n"
                "CIBIL score range: 300 to 900.\n"
                "\n"
                "Low risk: CIBIL score >= 750.\n"
                "Medium risk: CIBIL score 650-749.\n"
                "High risk: CIBIL score < 650.\n"
                "\n"
                "No credit history (score -1 / NA / NH) → MEDIUM (insufficient data, not a red flag by itself).\n"
                "Any loan default, settlement, or write-off on record → HIGH, regardless of numeric score."
            ),
            metadata={"source": "financial", "topic": "cibil_score", "inferred": True}
        ),





# ══════════════════════════════════════════════════════════════════════════════
    ]


# COLLECTION 2 — MEDICAL  (Pages 24-26 criteria + 36-47 impairments)
# ══════════════════════════════════════════════════════════════════════════════
def build_medical_chunks():
    return [

        # ── BLOOD PRESSURE — CONSOLIDATED ───────────────────────────────────
        Document(
            page_content=(
                "Blood Pressure Classification — All Risk Classes\n"
                "Age 18–55:\n"
                "Low risk: BP below 130/80.\n"
                "Medium risk: BP 130–145 / 80–90.\n"
                "High risk: BP above 145/90.\n"
                "\n"
                "Age 56+:\n"
                "Low risk: BP below 140/85.\n"
                "Medium risk: BP 140–150 / 85–92.\n"
                "High risk: BP above 150/92."
            ),
            metadata={"source": "medical", "topic": "blood_pressure"}
        ),

        # ── CHOLESTEROL & LIPID PROFILE — CONSOLIDATED ───────────────────────
        Document(
            page_content=(
                "Cholesterol and Lipid Profile — All Risk Classes\n"
                "Age 18–55:\n"
                "Low risk: total cholesterol below 200 mg/dL and cholesterol ratio below 4.5.\n"
                "Medium risk: total cholesterol 200–239 mg/dL or cholesterol ratio 4.5–7.0.\n"
                "High risk: total cholesterol 240 mg/dL or above, or cholesterol ratio 7.0 or above.\n"
                "\n"
                "Age 56+:\n"
                "Low risk: total cholesterol below 200 mg/dL and cholesterol ratio below 5.0.\n"
                "Medium risk: total cholesterol 200–239 mg/dL or cholesterol ratio 5.0–7.5.\n"
                "High risk: total cholesterol 240 mg/dL or above, or cholesterol ratio 7.5 or above."
            ),
            metadata={"source": "medical", "topic": "cholesterol_lipid_profile"}
        ),

        # ── BMI / BUILD — INDIAN CHART ──────────────────────────────────────
        Document(
            page_content=(
                "BMI and Build — Indian Population Chart\n"
                "Low risk: BMI 18.5–22.9.\n"
                "Medium risk: BMI 17.5–18.4 or 23.0–27.4.\n"
                "High risk: BMI below 17.5 or above 27.4."
            ),
            metadata={"source": "medical", "topic": "bmi_build"}
        ),

        # ── FAMILY HISTORY ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Family History — Death from Major Conditions\n"
                "Low risk: no cardiac or cancer death before age 60 in parents or siblings.\n"
                "Medium risk: one cardiac or cancer death before age 60 in parents or siblings.\n"
                "High risk: multiple deaths before age 60 from cardiac or cancer conditions."
            ),
            metadata={"source": "medical", "topic": "family_history"}
        ),

        # ── DIABETES ──────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Diabetes Mellitus — Risk Classification\n"
            "\n"
            "GENERAL SCREENING (no prior diabetes diagnosis):\n"
            "FBS < 100 mg/dL AND HbA1c < 5.7% → LOW\n"
            "FBS 100-125 mg/dL OR HbA1c 5.7-6.4% → MEDIUM\n"
            "FBS >= 126 mg/dL OR HbA1c >= 6.5% → HIGH (new diabetes diagnosis)\n"
            "\n"
            "TYPE II DIABETES (diagnosed, on treatment):\n"
            "HbA1c < 7.0% → MEDIUM (well controlled)\n"
            "HbA1c 7.0-8.9% → HIGH\n"
            "HbA1c 9.0-10.0% → HIGH\n"
            "HbA1c > 10.0% → HIGH\n"
            "\n"
            "TYPE I DIABETES (diagnosed): any HbA1c → HIGH\n"
            "\n"
            "Gestational diabetes (pregnant) → MEDIUM (postpone)\n"
            "History of gestational diabetes → MEDIUM\n"
            "Diabetes + Peripheral Vascular Disease → HIGH\n"
            "Diabetes + CKD (eGFR<30) → HIGH\n"
            "\n"
            "See Comorbidities for diabetes combined with other conditions."
        ),
        metadata={"source": "medical", "topic": "diabetes"}
    ),

    # ── HYPERTENSION ──────────────────────────────────────────────────────
    Document(
        page_content=(
            "Hypertension — Risk Classification\n"
            "\n"
            "BP <= 130/80 (on meds, well controlled) → MEDIUM (best-controlled tier, consider credit)\n"
            "BP 130/80 - 140/90 (on meds, controlled) → MEDIUM\n"
            "BP 140/90 - 160/100 (on meds, uncontrolled) → HIGH\n"
            "BP > 160/100 consistently → HIGH (severe uncontrolled)\n"
            "Hypertension + high BMI or diabetes → HIGH"
        ),
        metadata={"source": "medical", "topic": "hypertension", "inferred": True}
    ),

    # ── MYOCARDIAL INFARCTION ─────────────────────────────────────────────
    Document(
        page_content=(
            "Myocardial Infarction (Heart Attack) — Risk Classification\n"
            "\n"
            "Any CAD history (angina, MI, stent, bypass) → HIGH, regardless of age or time since event"
        ),
        metadata={"source": "medical", "topic": "myocardial_infarction"}
    ),

    # ── ANGINA ────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Angina — Risk Classification\n"
            "\n"
            "Stable or Unstable Angina → HIGH, regardless of age\n"
            "Stress-induced Angina, negative stress test / no ST depression >1mm → MEDIUM"
        ),
        metadata={"source": "medical", "topic": "angina", "inferred": True}
    ),

    # ── HEART FAILURE & ARRHYTHMIA ────────────────────────────────────────
    Document(
        page_content=(
            "Heart Failure and Arrhythmia — Risk Classification\n"
            "\n"
            "Congestive Heart Failure (any diagnosis) → HIGH\n"
            "\n"
            "AFib (paroxysmal or permanent) → HIGH"
        ),
        metadata={"source": "medical", "topic": "heart_failure_arrhythmia"}
    ),

    # ── FAMILY HISTORY — CAD ──────────────────────────────────────────────
    Document(
        page_content=(
            "Family History — Coronary Artery Disease — Risk Classification\n"
            "\n"
            "Father CAD death before age 60 → MEDIUM, regardless of applicant's own workup results"
        ),
        metadata={"source": "medical", "topic": "family_history_cad"}
    ),

    # ── KIDNEY FUNCTION ───────────────────────────────────────────────────
    Document(
        page_content=(
            "Kidney Function — Risk Classification\n"
            "\n"
            "Creatinine < 1.2 mg/dL → LOW\n"
            "Creatinine 1.3-1.7 mg/dL → MEDIUM\n"
            "Creatinine >= 1.8 mg/dL → HIGH\n"
            "\n"
            "eGFR >= 90 → LOW\n"
            "eGFR 60-89 → MEDIUM\n"
            "eGFR 45-59 → MEDIUM\n"
            "eGFR 30-44 → HIGH\n"
            "eGFR < 30 → HIGH\n"
            "\n"
            "Acute Nephritis → HIGH\n"
            "Chronic Nephritis, eGFR >= 60 → MEDIUM\n"
            "Chronic Nephritis, eGFR < 60 → HIGH\n"
            "Polycystic Kidney Disease (any function status) → HIGH\n"
            "Renal Failure (dialysis/transplant) → HIGH\n"
            "Kidney Transplant, stable → HIGH"
        ),
        metadata={"source": "medical", "topic": "kidney_function"}
    ),

    # ── LIVER FUNCTION ────────────────────────────────────────────────────
    Document(
        page_content=(
            "Liver Function — Risk Classification\n"
            "\n"
            "AST/ALT < 1x normal → LOW\n"
            "AST/ALT 1-3x normal → MEDIUM\n"
            "AST/ALT >= 3x normal → HIGH\n"
            "\n"
            "Bilirubin < 1.2 mg/dL → LOW\n"
            "Bilirubin 1.2-2.0 mg/dL → MEDIUM\n"
            "Bilirubin > 2.0 mg/dL → HIGH\n"
            "\n"
            "Cirrhosis (any stage) → HIGH\n"
            "Chronic Hepatitis B/C → HIGH\n"
            "Acute Hepatitis (active or after recovery) → HIGH\n"
            "Fatty Liver Disease (NAFLD), mild to moderate → MEDIUM\n"
            "Fatty Liver Disease, advanced fibrosis → HIGH\n"
            "Alcohol-related Liver Disease → HIGH"
        ),
        metadata={"source": "medical", "topic": "liver_function"}
    ),

    # ── THYROID ───────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Thyroid Function — Risk Classification\n"
            "\n"
            "GENERAL SCREENING (no prior thyroid diagnosis):\n"
            "TSH 0.5-4.0 mIU/L → LOW\n"
            "TSH 4.1-10 mIU/L → MEDIUM\n"
            "TSH > 10 mIU/L → HIGH\n"
            "TSH < 0.5 mIU/L → MEDIUM (subclinical hyperthyroid)\n"
            "\n"
            "DIAGNOSED CONDITIONS (on treatment):\n"
            "Hypothyroidism, controlled (TSH 0.5-4.0 on levothyroxine) → MEDIUM\n"
            "Hypothyroidism, uncontrolled (TSH>10) → HIGH\n"
            "Hashimoto's, controlled (TSH 0.5-4.0) → MEDIUM\n"
            "Hashimoto's, uncontrolled → HIGH\n"
            "\n"
            "Hyperthyroidism, controlled (TSH 0.5-4.0 on antithyroid drugs) → MEDIUM\n"
            "Graves' recovered, no complications → MEDIUM\n"
            "Active hyperthyroidism, uncontrolled (TSH < 0.5) → HIGH\n"
            "\n"
            "Goiter, euthyroid (TSH 0.5-4.0) → MEDIUM\n"
            "Goiter, abnormal function (TSH outside 0.5-4.0) → HIGH"
        ),
        metadata={"source": "medical", "topic": "thyroid", "inferred": True}
    ),

    # ── CBC ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Complete Blood Count (CBC) — Risk Classification\n"
            "\n"
            "Hemoglobin normal (men>=13.5, women>=12.0 g/dL) → LOW\n"
            "Hemoglobin mild anemia (women 11.0-12.9 / men 12-13.4) → MEDIUM\n"
            "Hemoglobin moderate anemia (8.0-10.9 g/dL) → HIGH\n"
            "Hemoglobin severe anemia (<8.0 g/dL) → HIGH\n"
            "\n"
            "WBC 4.5-11.0 x10^3/uL → LOW\n"
            "WBC 11.1-15 x10^3/uL → MEDIUM\n"
            "WBC > 15 x10^3/uL → HIGH\n"
            "WBC < 4.5 x10^3/uL → MEDIUM\n"
            "\n"
            "Platelets 150-400 x10^3/uL → LOW\n"
            "Platelets 100-150 x10^3/uL → MEDIUM\n"
            "Platelets < 100 x10^3/uL → HIGH\n"
            "Platelets > 400 x10^3/uL → MEDIUM\n"
            "\n"
            "Sickle Cell Disease → HIGH\n"
            "Sickle Cell Trait → MEDIUM\n"
            "Aplastic Anemia → HIGH\n"
            "Hemophilia → HIGH\n"
            "Thalassemia major → HIGH\n"
            "Thalassemia minor → MEDIUM"
        ),
        metadata={"source": "medical", "topic": "cbc_blood_count"}
    ),

    # ── INFECTIOUS DISEASES ───────────────────────────────────────────────
    Document(
        page_content=(
            "Infectious Diseases — Risk Classification\n"
            "\n"
            "HIV negative → LOW\n"
            "HIV positive → HIGH\n"
            "\n"
            "HBsAg negative → LOW\n"
            "HBsAg positive → HIGH\n"
            "\n"
            "Anti-HCV negative → LOW\n"
            "Anti-HCV positive → HIGH\n"
            "\n"
            "VDRL/RPR negative → LOW\n"
            "VDRL/RPR positive, treated → MEDIUM\n"
            "VDRL/RPR positive, untreated → HIGH\n"
            "\n"
            "Malaria/Dengue history, resolved → MEDIUM\n"
            "Malaria/Dengue currently positive → HIGH (postpone)"
        ),
        metadata={"source": "medical", "topic": "infectious_disease"}
    ),

    # ── ECG ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Electrocardiogram (ECG) — Risk Classification\n"
            "\n"
            "Normal sinus rhythm, HR 60-100 bpm → LOW\n"
            "Sinus bradycardia (HR 50-59) → MEDIUM\n"
            "Sinus tachycardia (HR 101-120) → MEDIUM\n"
            "\n"
            "AFib (paroxysmal, permanent, or new-onset) → HIGH\n"
            "\n"
            "RBBB → MEDIUM\n"
            "LBBB (any duration) → HIGH\n"
            "1st-degree AV block → MEDIUM\n"
            "2nd-degree+ AV block → HIGH\n"
            "\n"
            "ST elevation/depression → HIGH\n"
            "T wave inversion → MEDIUM\n"
            "ST depression > 1mm on stress test → HIGH\n"
            "\n"
            "LVH → MEDIUM\n"
            "LVH + strain pattern → HIGH"
        ),
        metadata={"source": "medical", "topic": "ecg", "inferred": True}
    ),

    # ── SPO2 & PULSE ──────────────────────────────────────────────────────
    Document(
        page_content=(
            "SpO2 and Pulse Rate — Risk Classification\n"
            "\n"
            "SpO2 >= 95% → LOW\n"
            "SpO2 93-94% → MEDIUM\n"
            "SpO2 < 93% → HIGH\n"
            "\n"
            "HR 55-100 bpm → LOW\n"
            "HR 50-54 bpm → MEDIUM\n"
            "HR 101-110 bpm → MEDIUM\n"
            "HR < 50 bpm → HIGH\n"
            "HR > 110 bpm → HIGH"
        ),
        metadata={"source": "medical", "topic": "respiratory_vitals", "inferred": True}
    ),

    # ── URINALYSIS ────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Urinalysis — Risk Classification\n"
            "\n"
            "Glucose negative → LOW\n"
            "Glucose trace-1+ → MEDIUM\n"
            "Glucose >= 2+ → HIGH\n"
            "\n"
            "Protein negative → LOW\n"
            "Protein trace-1+ → MEDIUM\n"
            "Protein >= 2+ → HIGH\n"
            "Microalbuminuria 30-300 mg/day → MEDIUM\n"
            "\n"
            "Ketones negative → LOW\n"
            "Ketones positive → HIGH\n"
            "\n"
            "RBC negative/rare → LOW\n"
            "RBC 1-5/field → MEDIUM\n"
            "RBC > 5/field → HIGH\n"
            "\n"
            "WBC (pyuria) negative/rare → LOW\n"
            "WBC (pyuria) present → MEDIUM"
        ),
        metadata={"source": "medical", "topic": "urinalysis"}
    ),

    # ── SLEEP APNEA ───────────────────────────────────────────────────────
    Document(
        page_content=(
            "Sleep Apnea — Risk Classification\n"
            "\n"
            "AHI < 5 → LOW\n"
            "AHI 5-15 (mild) → MEDIUM\n"
            "AHI 15-30 (moderate) → MEDIUM\n"
            "AHI > 30 (severe) → HIGH\n"
            "\n"
            "CPAP compliant = >=4 hrs/night on >=70% of nights\n"
            "CPAP non-compliant = <4 hrs/night or <5 nights/week\n"
            "\n"
            "Untreated, any severity → HIGH\n"
            "Mild, no CPAP needed → MEDIUM\n"
            "Moderate, CPAP compliant (>=4h/night, >=70% nights) → MEDIUM\n"
            "Moderate, non-compliant (<4h/night or <5 nights/week) → HIGH\n"
            "Severe, CPAP compliant → HIGH\n"
            "\n"
            "Central Sleep Apnea → HIGH\n"
            "\n"
            "See Comorbidities for sleep apnea combined with other conditions."
        ),
        metadata={"source": "medical", "topic": "sleep_apnea"}
    ),

    # ── STROKE ────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Stroke — Risk Classification\n"
            "\n"
            "Stroke < 5 years → HIGH\n"
            "Stroke > 5 years, no recurrence → MEDIUM\n"
            "(Minimum 6 months post-stroke to be insurable)"
        ),
        metadata={"source": "medical", "topic": "stroke"}
    ),

    # ── TIA ───────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Transient Ischemic Attack (TIA) — Risk Classification\n"
            "\n"
            "TIA single, > 6 months ago → MEDIUM\n"
            "TIA multiple (2-3) within 1 year → HIGH\n"
            "TIA recent (< 6 months) → HIGH"
        ),
        metadata={"source": "medical", "topic": "tia"}
    ),

    # ── MULTIPLE SCLEROSIS ────────────────────────────────────────────────
    Document(
        page_content=(
            "Multiple Sclerosis — Risk Classification\n"
            "\n"
            "MS, EDSS 0-3.5 (mild, stable) → MEDIUM\n"
            "MS, EDSS >= 4.0, or frequent relapses → HIGH"
        ),
        metadata={"source": "medical", "topic": "multiple_sclerosis", "inferred": True}
    ),

    # ── PARKINSON'S DISEASE ───────────────────────────────────────────────
    Document(
        page_content=(
            "Parkinson's Disease — Risk Classification\n"
            "\n"
            "Parkinson's, Hoehn & Yahr Stage I-II (mild/early) → MEDIUM\n"
            "Parkinson's, Hoehn & Yahr Stage III or higher (moderate to advanced) → HIGH"
        ),
        metadata={"source": "medical", "topic": "parkinsons_disease", "inferred": True}
    ),

    # ── EPILEPSY ──────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Epilepsy — Risk Classification\n"
            "\n"
            "Epilepsy well-controlled (no seizures 2+ yrs) → MEDIUM\n"
            "Epilepsy with any breakthrough seizures (occasional or frequent) → HIGH"
        ),
        metadata={"source": "medical", "topic": "epilepsy"}
    ),

    # ── CANCER ────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Cancer and Malignancy — Risk Classification\n"
            "\n"
            "Most solid malignancies, within 5yr postponement → HIGH\n"
            "Most solid malignancies, after 5yr clear → MEDIUM\n"
            "Lymphoma, within 2-3yr postponement → HIGH\n"
            "Lymphoma, after postponement, in remission → MEDIUM\n"
            "Acute Leukemia, within 5-10yr postponement → HIGH\n"
            "Acute Leukemia, after 10yr clear → MEDIUM\n"
            "Chronic Leukemia controlled, within 2-3yr → HIGH\n"
            "Chronic Leukemia controlled, after 2-3yr, in remission → MEDIUM\n"
            "\n"
            "Basal Cell Carcinoma, fully resolved → MEDIUM\n"
            "Superficial Squamous Cell Carcinoma → MEDIUM\n"
            "\n"
            "Melanoma Stage 1, after 2yr clear period → MEDIUM\n"
            "Melanoma Stage 2-3 → HIGH\n"
            "\n"
            "Currently under treatment, or within 12 months post-treatment → HIGH\n"
            "\n"
            "Any recurrence or metastatic disease → HIGH"
        ),
        metadata={"source": "medical", "topic": "cancer", "inferred": True}
    ),

    # ── COMORBIDITIES ─────────────────────────────────────────────────────
    Document(
        page_content=(
            "Comorbidities — Compound Risk Classification\n"
            "\n"
            "Diabetes + Hypertension → HIGH\n"
            "Diabetes + BMI>30 + Age>50 → HIGH\n"
            "Diabetes + Albuminuria + Low eGFR → HIGH\n"
            "Hypertension + High cholesterol + LVH → HIGH\n"
            "Tobacco + CAD history → HIGH\n"
            "Sleep Apnea + Hypertension + Obesity (BMI>35) → HIGH\n"
            "Hepatitis B/C + Alcohol use history → HIGH\n"
            "\n"
            "Note: multiple conditions combine non-additively — "
            "always escalates toward HIGH, never simply averages."
        ),
        metadata={"source": "medical", "topic": "comorbidities"}
    ),

]


# ══════════════════════════════════════════════════════════════════════════════
# COLLECTION 3 — LIFESTYLE (Occupation, Avocation, Habits — Indian Context)
# ══════════════════════════════════════════════════════════════════════════════
def build_lifestyle_chunks():
    return [
 
        # ── IT / SOFTWARE ────────────────────────────────────────────────────────
        Document(
            page_content=(
                "IT / Software Occupations — Risk Classification\n"
                "\n"
                "IT / Software Engineer / Developer / Programmer / IT Manager → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "it_software", "risk_level": "LOW"}
        ),

        # ── HEALTHCARE PROFESSIONALS ─────────────────────────────────────────────
        Document(
            page_content=(
                "Healthcare Professionals, Hospital/Clinic-Based — Risk Classification\n"
                "\n"
                "Doctor / Physician / Surgeon / Dentist (hospital/clinic-based) → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "healthcare_professional", "risk_level": "LOW"}
        ),

        # ── EDUCATION ────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Education Occupations — Risk Classification\n"
                "\n"
                "Teacher / Professor / Educator / Principal → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "education", "risk_level": "LOW"}
        ),

        # ── FINANCE & ACCOUNTING ─────────────────────────────────────────────────
        Document(
            page_content=(
                "Finance and Accounting Occupations — Risk Classification\n"
                "\n"
                "Chartered Accountant / Accountant / Finance Manager / CFO → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "finance_accounting", "risk_level": "LOW"}
        ),

        # ── LEGAL ────────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Legal Occupations — Risk Classification\n"
                "\n"
                "Lawyer / Advocate / Legal Consultant → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "legal", "risk_level": "LOW"}
        ),

        # ── BANKING & INVESTMENT ─────────────────────────────────────────────────
        Document(
            page_content=(
                "Banking and Investment Occupations — Risk Classification\n"
                "\n"
                "Banker / Finance Executive / Investment Banker / Fund Manager → LOW"
            ),
            metadata={"source": "lifestyle", "topic": "banking_investment", "risk_level": "LOW"}
        ),

        # ── GOVERNMENT OFFICER ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Government Officer — Risk Classification\n"
                "\n"
                "Government Officer, office-based (IAS/IPS/clerk/admin) → LOW\n"
                "Government Officer, field-based (forest ranger, excise field officer) → MEDIUM"
            ),
            metadata={"source": "lifestyle", "topic": "government_officer"}
        ),

        # ── ARCHITECT ────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Architect — Risk Classification\n"
                "\n"
                "Architect, office-based (design only) → LOW\n"
                "Architect, site supervisor on construction → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "architect"}
        ),

        # ── ENGINEER ─────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Engineer — Risk Classification\n"
                "\n"
                "Engineer, office-based (civil/electrical/mechanical design) → LOW\n"
                "Engineer, site engineer on construction/mining → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "engineer"}
        ),
 
        # ── DRIVERS ──────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Drivers — Risk Classification\n"
                "\n"
                "Taxi / Auto / Cab / Private vehicle driver → MEDIUM\n"
                "Local city bus driver → MEDIUM\n"
                "Inter-city bus driver → MEDIUM\n"
                "Long-haul truck driver → HIGH\n"
                "Tanker / Hazmat driver → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "drivers"}
        ),
 
        # ── FACTORY / MANUFACTURING ──────────────────────────────────────────────
        Document(
            page_content=(
                "Factory / Manufacturing Workers — Risk Classification\n"
                "\n"
                "Textile / Garment factory worker → MEDIUM\n"
                "Electronics assembly worker → MEDIUM\n"
                "Food processing worker → MEDIUM\n"
                "Chemical factory worker → HIGH\n"
                "Steel / Iron plant worker → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "factory_manufacturing"}
        ),
 
        # ── MINING ───────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Mining Occupations — Risk Classification\n"
                "\n"
                "Underground coal / mineral miner → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "mining", "risk_level": "HIGH"}
        ),

        # ── HIGH-RISE CONSTRUCTION ───────────────────────────────────────────────
        Document(
            page_content=(
                "High-Rise Construction Occupations — Risk Classification\n"
                "\n"
                "High-rise construction / scaffolding / steelwork worker → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "construction_highrise", "risk_level": "HIGH"}
        ),

        # ── CHEMICAL / HAZMAT ────────────────────────────────────────────────────
        Document(
            page_content=(
                "Chemical and Hazmat Occupations — Risk Classification\n"
                "\n"
                "Chemical plant operator / hazmat handler → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "chemical_hazmat", "risk_level": "HIGH"}
        ),

        # ── AVIATION ─────────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Aviation Occupations — Risk Classification\n"
                "\n"
                "Commercial airline pilot / co-pilot / flight engineer → HIGH\n"
                "Private / recreational pilot → MEDIUM"
            ),
            metadata={"source": "lifestyle", "topic": "aviation"}
        ),

        # ── OFFSHORE OIL & GAS ───────────────────────────────────────────────────
        Document(
            page_content=(
                "Offshore Oil and Gas Occupations — Risk Classification\n"
                "\n"
                "Offshore oil/gas rig worker / platform operator → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "offshore_oil_gas", "risk_level": "HIGH"}
        ),
 
        # ── ARMED FORCES ─────────────────────────────────────────────────────────
        Document(
            page_content=(
                "Armed Forces / Military / Paramilitary — Risk Classification\n"
                "\n"
                "Rear-line / administrative role, peacetime → MEDIUM\n"
                "Combat-trained soldier, peacetime posting → HIGH\n"
                "Recently returned from combat (< 2 years) → HIGH\n"
                "Active deployment to conflict zone → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "armed_forces"}
        ),
 
        # ── AVOCATIONS: SCUBA DIVING ────────────────────────────────────────────
        Document(
            page_content=(
                "Avocation — Scuba Diving — Risk Classification\n"
                "\n"
                "Snorkeling (no tank) → LOW\n"
                "Recreational scuba < 18m depth, certified, < 10 dives/year → LOW\n"
                "Recreational scuba 18-30m depth → MEDIUM\n"
                "Recreational scuba 30-40m depth → MEDIUM\n"
                "Recreational scuba > 40m depth → HIGH\n"
                "Cave diving → HIGH\n"
                "> 10 dives/year at depth → escalate one level"
            ),
            metadata={"source": "lifestyle", "topic": "avocation_scuba", "is_avocation": True}
        ),
 
        # ── AVOCATIONS: TREKKING / MOUNTAINEERING ───────────────────────────────
        Document(
            page_content=(
                "Avocation — Trekking / Mountaineering — Risk Classification\n"
                "\n"
                "Hill trekking (< 3000m altitude, established trails) → LOW\n"
                "High altitude trekking (3000-4500m, snow possible) → MEDIUM\n"
                "Technical rock climbing (4500-6000m, ropes needed) → MEDIUM\n"
                "High altitude mountaineering (6000-7500m, oxygen needed) → HIGH\n"
                "Extreme altitude (> 8000m, e.g. Everest) → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "avocation_mountaineering", "is_avocation": True}
        ),
 
        # ── AVOCATIONS: MOTORSPORTS ──────────────────────────────────────────────
        Document(
            page_content=(
                "Avocation — Motorsports / Racing — Risk Classification\n"
                "\n"
                "Recreational track driving (not racing) → MEDIUM\n"
                "Professional car / motorcycle / boat racing → HIGH\n"
                "Rally driving (professional) → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "avocation_racing", "is_avocation": True}
        ),

        # ── AVOCATIONS: PRIVATE AVIATION, PARACHUTING, HANG GLIDING, BALLOONING ──
        Document(
            page_content=(
                "Avocation — Private Aviation, Parachuting, Hang Gliding, Hot Air Ballooning — Risk Classification\n"
                "\n"
                "Private Aviation:\n"
                "Passenger only, not a licensed pilot → LOW\n"
                "Licensed private pilot, fixed-wing, < 100 hours/year → MEDIUM\n"
                "Licensed private pilot, >= 100 hours/year, aerobatic, or experimental aircraft → HIGH\n"
                "\n"
                "Parachuting:\n"
                "No participation → LOW\n"
                "Tandem jump, occasional (< 5 jumps/year) → MEDIUM\n"
                "Licensed solo skydiver, regular (>= 5 jumps/year) → HIGH\n"
                "BASE jumping → HIGH\n"
                "\n"
                "Hang Gliding:\n"
                "No participation → LOW\n"
                "Recreational, certified, < 20 flights/year → MEDIUM\n"
                "Competitive, or >= 20 flights/year → HIGH\n"
                "\n"
                "Hot Air Ballooning:\n"
                "No participation → LOW\n"
                "Passenger on licensed commercial ride → LOW\n"
                "Licensed balloon pilot → MEDIUM"
            ),
            metadata={"source": "lifestyle", "topic": "avocation_aviation_airborne", "is_avocation": True}
        ),

        # ── SMOKING & NICOTINE ───────────────────────────────────────────────
        Document(
            page_content=(
                "Smoking, Tobacco, and Nicotine Use — Risk Classification\n"
                "Low risk: nicotine-free for 36+ months; occasional cigar use may be acceptable with negative test.\n"
                "Medium risk: nicotine-free for 12–36 months, or occasional cannabis use.\n"
                "High risk: current nicotine use, nicotine-free for less than 12 months, regular cannabis use, or positive drug screen."
            ),
            metadata={"source": "lifestyle", "topic": "smoking_nicotine"}
        ),

        # ── ALCOHOL USE ───────────────────────────────────────────────────────
        Document(
            page_content=(
                "Alcohol Use — Risk Classification\n"
                "Based on drinking frequency, in occasions per week.\n"
                "\n"
                "0 times per week (non-drinker / abstinent) → LOW\n"
                "1-2 times per week → MEDIUM\n"
                "More than 2 times per week → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "alcohol_use"}
        ),

        # ── DRUNK DRIVING ─────────────────────────────────────────────────────
        Document(
            page_content=(
                "Drunk Driving — Risk Classification\n"
                "\n"
                "Caught drunk driving 0 times in last 12 months → LOW\n"
                "Caught drunk driving 1-2 times in last 12 months → MEDIUM\n"
                "Caught drunk driving more than 2 times in last 12 months → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "drunk_driving"}
        ),

        # ── ACCIDENTS WHILE DRIVING ──────────────────────────────────────────
        Document(
            page_content=(
                "Accidents While Driving — Risk Classification\n"
                "\n"
                "0 accidents in last 1 year → LOW\n"
                "1 accident in last 1 year → MEDIUM\n"
                "More than 1 accident in last 1 year → HIGH"
            ),
            metadata={"source": "lifestyle", "topic": "driving_accidents"}
        ),

    ]

# ══════════════════════════════════════════════════════════════════════════════
# HELPER — Delete a collection's old embeddings before rebuilding it
# ══════════════════════════════════════════════════════════════════════════════
def _reset_collection(name):
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(name=name)
        print(f"  🗑️  Deleted old '{name}'")
    except Exception:
        pass  # collection didn't exist yet — nothing to delete


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — Rebuild all 3 collections from scratch (safe to rerun anytime)
# ══════════════════════════════════════════════════════════════════════════════
def build_all_collections():
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print("Building collection_financial...")
    _reset_collection("collection_financial")
    fin = build_financial_chunks()
    Chroma.from_documents(fin, embeddings,
                          collection_name="collection_financial",
                          persist_directory=CHROMA_DIR)
    print(f"  ✅ {len(fin)} chunks")

    print("Building collection_medical...")
    _reset_collection("collection_medical")
    med = build_medical_chunks()
    Chroma.from_documents(med, embeddings,
                          collection_name="collection_medical",
                          persist_directory=CHROMA_DIR)
    print(f"  ✅ {len(med)} chunks")

    print("Building collection_lifestyle...")
    _reset_collection("collection_lifestyle")
    life = build_lifestyle_chunks()
    Chroma.from_documents(life, embeddings,
                          collection_name="collection_lifestyle",
                          persist_directory=CHROMA_DIR)
    print(f"  ✅ {len(life)} chunks")

    print(f"\n🎉 Done! ChromaDB at: {CHROMA_DIR}")


if __name__ == "__main__":
    build_all_collections()