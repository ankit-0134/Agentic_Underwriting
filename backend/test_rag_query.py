"""
Test RAG embeddings and queries
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv(Path(__file__).parent / ".env")

CHROMA_DIR = r"C:\Users\ankit\Documents\Underwriting\backend\chroma_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Test Financial Collection
print("=" * 70)
print("🏦 FINANCIAL COLLECTION QUERIES")
print("=" * 70)

fin_db = Chroma(
    collection_name="collection_financial",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

queries = [
    "applicant age band 20-40 income 10 lakhs sum assured 2.5 crores",
    "applicant age band 56-65 sum assured 15 times annual income",
    "total monthly EMI 40000 net salary 100000",
    "credit score 620 loan default history",
]

for q in queries:
    print(f"\n📝 Query: {q}")
    results = fin_db.similarity_search(q, k=2)
    for i, doc in enumerate(results, 1):
        topic = doc.metadata.get("topic", "N/A")
        print(f"  [{i}] Topic: {topic}")
        print(f"      {doc.page_content[:150]}...\n")

# Test Medical Collection
print("\n" + "=" * 70)
print("🏥 MEDICAL COLLECTION QUERIES")
print("=" * 70)

med_db = Chroma(
    collection_name="collection_medical",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

queries = [
    "diabetes HbA1c 7.5 blood sugar control",
    "BMI 28 overweight risk assessment",
    "father died heart attack age 55 family history",
    "heart attack myocardial infarction history",
    "chest pain angina stress test",
    "atrial fibrillation irregular heartbeat",
    "stroke brain attack recovery period",
    "epilepsy seizure disorder control",
    "chronic kidney disease eGFR nephritis",
    "fatty liver hepatitis B carrier",
    "thyroid TSH hypothyroidism",
    "leukemia blood cancer treatment",
    "sleep apnea CPAP compliance",
]

for q in queries:
    print(f"\n📝 Query: {q}")
    results = med_db.similarity_search(q, k=2)
    for i, doc in enumerate(results, 1):
        topic = doc.metadata.get("topic", "N/A")
        print(f"  [{i}] Topic: {topic}")
        print(f"      {doc.page_content[:150]}...\n")

# Test Lifestyle Collection
print("\n" + "=" * 70)
print("🚬 LIFESTYLE COLLECTION QUERIES")
print("=" * 70)

life_db = Chroma(
    collection_name="collection_lifestyle",
    embedding_function=embeddings,
    persist_directory=CHROMA_DIR
)

queries = [
    "software engineer developer IT professional",
    "underground coal miner mining hazard",
    "commercial airline pilot aviation",
    "architect site supervisor construction",
    "scuba diving depth certification",
    "smoking tobacco nicotine use",
    "drunk driving DUI conviction",
]

for q in queries:
    print(f"\n📝 Query: {q}")
    results = life_db.similarity_search(q, k=2)
    for i, doc in enumerate(results, 1):
        topic = doc.metadata.get("topic", "N/A")
        risk = doc.metadata.get("risk_level", "N/A")
        print(f"  [{i}] Topic: {topic} | Risk: {risk}")
        print(f"      {doc.page_content[:150]}...\n")

print("\n✅ RAG Query Test Complete!")
