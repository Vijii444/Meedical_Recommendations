import pandas as pd

print("--- Starting Dataset Cleaning ---")

# 1. Clean dataset.csv (Main Training Data) inside the data folder
df_dataset = pd.read_csv("data/dataset.csv")
df_dataset.columns = df_dataset.columns.str.strip()
df_dataset["Disease"] = df_dataset["Disease"].str.strip()
for col in df_dataset.columns:
  if df_dataset[col].dtype == "object":
    df_dataset[col] = df_dataset[col].str.strip()
df_dataset.to_csv("data/cleaned_dataset.csv", index=False)
print("✔ cleaned_dataset.csv ready.")

# 2. Clean symptom_Description.csv
df_desc = pd.read_csv("data/symptom_Description.csv")
df_desc.columns = df_desc.columns.str.strip()
df_desc["Disease"] = df_desc["Disease"].str.strip()
df_desc["Description"] = df_desc["Description"].str.strip()
df_desc.to_csv("data/cleaned_description.csv", index=False)
print("✔ cleaned_description.csv ready.")

# 3. Clean symptom_precaution.csv
df_prec = pd.read_csv("data/symptom_precaution.csv")
df_prec.columns = df_prec.columns.str.strip()
df_prec["Disease"] = df_prec["Disease"].str.strip()
for col in df_prec.columns:
  if df_prec[col].dtype == "object":
    df_prec[col] = df_prec[col].str.strip()
df_prec.to_csv("data/cleaned_precaution.csv", index=False)
print("✔ cleaned_precaution.csv ready.")

# 4. Clean Symptom-severity.csv
df_sev = pd.read_csv("data/Symptom-severity.csv")
df_sev.columns = df_sev.columns.str.strip()
df_sev["Symptom"] = df_sev["Symptom"].str.strip()
df_sev.to_csv("data/cleaned_severity.csv", index=False)
print("✔ cleaned_severity.csv ready.")

print("\nSUCCESS: All files successfully cleaned and saved!")