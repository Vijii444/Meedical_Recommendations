# 🏥 HealthPulse AI: Secure Grounded AI Healthcare Recommendation Portal

An enterprise-grade, full-stack AI healthcare web application featuring machine learning symptom classification, secure PostgreSQL data retrieval, dynamic dataset symptom resolution, and grounded Groq LLM clinical consultations.

---

## 🌟 Key Features

- **Dynamic Natural Language Processing:** Automatically resolves everyday symptom descriptions or direct disease names (e.g., *"malaria"*, *"fungal infection"*) using dynamic CSV mapping on startup, preventing 400 bad requests.
- **Machine Learning Classification:** Powered by a trained Random Forest Classifier mapping symptom feature vectors to accurate disease predictions.
- **Secure Relational Database:** Integrates with PostgreSQL (`healthcare_db`) to fetch verified pathology descriptions, official precautions, and symptom severity weights.
- **Anti-Hallucination Grounded AI Layer:** Utilizes Groq LLM integration (`openai/gpt-oss-20b`) with strict system prompts to generate structured 7-day recovery roadmaps and professional clinician summaries based *strictly* on database facts.
- **Emerald Frosted Glass Frontend:** A responsive, single-file interactive UI (`index.html`) featuring tabbed navigation, mode toggles (Patient vs. Clinician view), and custom background support built with Tailwind CSS.

---

## 🛠️ Technology Stack

- **Backend:** FastAPI, Python, Uvicorn, Pydantic, SQLAlchemy, Psycopg2
- **Machine Learning & Data:** Scikit-learn, Pandas, Joblib
- **Database:** PostgreSQL
- **AI / LLM:** Groq API (`openai/gpt-oss-20b`)
- **Frontend:** Tailwind CSS, Marked.js, Vanilla JavaScript, HTML5

---

## 📂 Project Structure

```text
Meedical_Recommendations/
│
├── Data/                      # Source CSV datasets & clean tables
│   ├── cleaned_dataset.csv
│   ├── cleaned_description.csv
│   ├── cleaned_precaution.csv
│   └── cleaned_severity.csv
│
├── models/                    # Trained ML artifacts
│   ├── model.pkl
│   └── feature_columns.pkl
│
├── background.jpg             # UI background asset
├── main.py                    # FastAPI application & dynamic routing backend
├── train_model.py             # Script to train Random Forest model
├── load_to_postgres.py        # Database migration script
├── index.html                 # Interactive single-file frontend
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
