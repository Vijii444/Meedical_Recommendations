import urllib.parse
import pandas as pd

# 1. Database Connection Details
DB_USER = "postgres"
DB_RAW_PASSWORD = "Hello@123"
DB_HOST = "localhost"
DB_PORT = "5432"
DB_NAME = "healthcare_db"

# Safely encode the password so special characters like '@' don't break the URI
DB_PASSWORD = urllib.parse.quote_plus(DB_RAW_PASSWORD)

# Create the full connection URI string
db_uri = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

print("Connecting to PostgreSQL database...")

# 2. Load and push tables by passing the connection URI directly to pandas
try:
  print("Loading description data into database...")
  df_desc = pd.read_csv("data/cleaned_description.csv")
  df_desc.to_sql("descriptions", con=db_uri, if_exists="replace", index=False)

  print("Loading precaution data into database...")
  df_prec = pd.read_csv("data/cleaned_precaution.csv")
  df_prec.to_sql("precautions", con=db_uri, if_exists="replace", index=False)

  print("Loading severity data into database...")
  df_sev = pd.read_csv("data/cleaned_severity.csv")
  df_sev.to_sql("severity", con=db_uri, if_exists="replace", index=False)

  print(
      "\nSUCCESS: All reference tables successfully created and loaded into"
      " PostgreSQL!"
  )

except Exception as e:
  print(f"Error connecting or loading data into database: {e}")