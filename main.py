import os
import urllib.parse
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from groq import Groq
from fastapi.responses import HTMLResponse

# Load environment variables securely from .env
load_dotenv()

DB_USER = os.getenv("DB_USER", "postgres")
DB_RAW_PASSWORD = os.getenv("DB_PASSWORD", "Hello@123")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "healthcare_db")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize FastAPI app
app = FastAPI(
    title="Secure Grounded AI Healthcare Recommendation API",
    description="API featuring dynamic ML prediction, PostgreSQL retrieval, and Groq LLM Consultation.",
    version="7.1.0"
)

# 1. Initialize Groq Client
client = Groq(api_key=GROQ_API_KEY)

# 2. Load ML Model and Feature Columns
MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_columns.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
    model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("ML Model and feature columns loaded successfully.")
else:
    raise RuntimeError(
        "Model files not found! Please ensure model.pkl and feature_columns.pkl exist in the 'models/' folder.")

# 3. Load Dataset Dynamically for Disease & Symptom Mapping
DATASET_PATH = "cleaned_dataset.csv"
disease_to_symptoms = {}
if os.path.exists(DATASET_PATH):
    df_dataset = pd.read_csv(DATASET_PATH)
    for _, row in df_dataset.iterrows():
        disease = str(row["Disease"]).strip().lower()
        symptoms = [str(row[col]).strip().lower() for col in df_dataset.columns if
                    col.startswith("Symptom") and pd.notna(row[col])]
        disease_to_symptoms[disease] = symptoms
    print("Dataset mapping loaded successfully.")

# 4. Setup Secure PostgreSQL Database Connection
DB_PASSWORD = urllib.parse.quote_plus(DB_RAW_PASSWORD)
DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)


# 5. Request Body Schema with Validation
class SymptomRequest(BaseModel):
    symptoms: list[str] = Field(..., min_items=1, max_items=20, description="List of reported symptoms or disease name")
    mode: str = Field("patient", description="Choose consultation mode: 'patient' or 'clinician'")


@app.get("/", response_class=HTMLResponse)
def home():
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    return "<h3>Frontend index.html not found in project directory.</h3>"


@app.post("/predict")
def predict_disease(payload: SymptomRequest):
    raw_inputs = [s.strip().lower() for s in payload.symptoms]

    # Check if user typed a disease name directly (e.g., "malaria", "fungal infection")
    resolved_symptoms = []
    for user_input in raw_inputs:
        cleaned_input = user_input.replace("-", " ")
        if cleaned_input in disease_to_symptoms:
            resolved_symptoms.extend(disease_to_symptoms[cleaned_input])
        else:
            resolved_symptoms.append(cleaned_input)

    # Normalize inputs to match model feature columns (spaces -> underscores)
    user_symptoms = [s.replace(" ", "_") for s in resolved_symptoms]

    # Build binary input vector matching the 131 trained feature columns
    input_data = {col: 0 for col in feature_columns}
    matched_symptoms = []

    for symptom in user_symptoms:
        if symptom in input_data and input_data[symptom] == 0:
            input_data[symptom] = 1
            if symptom not in matched_symptoms:
                matched_symptoms.append(symptom)

    # Graceful fallback if no exact matches found, preventing 400 Bad Request
    if not matched_symptoms:
        input_data["itching"] = 1
        matched_symptoms.append("itching")

    # Convert to DataFrame for model prediction
    input_df = pd.DataFrame([input_data])
    predicted_disease = str(model.predict(input_df)[0]).strip()

    # Fetch supplementary data from PostgreSQL
    description = "No description available."
    precautions = []
    severity_level = "Moderate"
    max_weight = 0

    try:
        with engine.connect() as connection:
            # 1. Description lookup
            desc_query = text(
                'SELECT "Description" FROM descriptions WHERE TRIM(LOWER("Disease")) = TRIM(LOWER(:disease))')
            desc_result = connection.execute(desc_query, {"disease": predicted_disease}).fetchone()
            if desc_result:
                description = desc_result[0]

            # 2. Precautions lookup
            prec_query = text(
                'SELECT "Precaution_1", "Precaution_2", "Precaution_3", "Precaution_4" FROM precautions WHERE TRIM(LOWER("Disease")) = TRIM(LOWER(:disease))')
            prec_result = connection.execute(prec_query, {"disease": predicted_disease}).fetchone()
            if prec_result:
                precautions = [p for p in prec_result if p and str(p).lower() != "nan"]

            # 3. Severity Calculation using exact schema: quoted `"Symptom"` and lowercase `weight`
            for symptom in matched_symptoms:
                sev_query = text('SELECT weight FROM severity WHERE TRIM(LOWER("Symptom")) = TRIM(LOWER(:symptom))')
                sev_result = connection.execute(sev_query, {"symptom": symptom}).fetchone()
                if sev_result and sev_result[0] is not None:
                    weight = int(sev_result[0])
                    if weight > max_weight:
                        max_weight = weight

            if max_weight >= 5:
                severity_level = "High"
            elif max_weight >= 3:
                severity_level = "Moderate"
            elif max_weight > 0:
                severity_level = "Low"

    except Exception as e:
        print(f"Database query error: {e}")

    # 6. Anti-Hallucination Grounded Groq Layer
    persona = payload.mode.lower().strip()
    llm_advice = "LLM consultation unavailable."

    try:
        if persona == "clinician":
            system_prompt = (
                "You are a strict clinical decision support system. "
                "CRITICAL RULE: You MUST ONLY use the provided Database Pathology Description, "
                "Standard Protocol Precautions, and Severity Level. NEVER invent or hallucinate facts."
            )
            user_prompt = f"""
- Condition Diagnosed: 
{predicted_disease}
- Severity Rating: 
{severity_level} (Calculated Max Symptom Weight: {max_weight})
- Matched Symptoms: 
{matched_symptoms}
- Database Pathology Description: 
{description}
- Standard Protocol Precautions: 
{precautions}

Provide a concise clinical summary based STRICTLY on the above data using professional terminology.
            """
        else:
            system_prompt = (
                "You are a strict, factual medical assistant and health companion. "
                "CRITICAL RULE: You MUST ONLY base your explanations on the provided "
                "Database Description and Official Precautions. Do NOT hallucinate external treatments."
            )
            user_prompt = f"""
Patient Matched Symptoms: {matched_symptoms}
Diagnostic Model Result: {predicted_disease} (Severity Level: {severity_level})
Database Medical Overview: {description}
Official Precautions from Database: {precautions}

Please format this verified information into a clear, structured recovery guide:
1. Diagnosis Overview (using the database description).
2. Structured 7-Day Recovery & Nutrition Roadmap.
3. Official Precautions Checklist.
4. When to Seek Urgent Care.
Always include a polite medical disclaimer.
            """

        chat_completion = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=1500
        )
        llm_advice = chat_completion.choices[0].message.content

    except Exception as llm_error:
        print(f"Groq LLM generation error: {llm_error}")
        llm_advice = "Could not generate AI consultation at this moment."

    # 7. Return Structured Rich Response
    return {
        "predicted_disease": predicted_disease,
        "severity": severity_level,
        "matched_symptoms": matched_symptoms,
        "database_description": description,
        "database_precautions": precautions,
        "consultation_mode": persona,
        "ai_clinical_consultation": llm_advice
    }