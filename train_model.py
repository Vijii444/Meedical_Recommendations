import os
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

print("Loading cleaned training dataset...")
dataset_path = "data/cleaned_dataset.csv"

if not os.path.exists(dataset_path):
  print(
      f"Error: Could not find {dataset_path}. Please check your 'data/' folder."
  )
else:
  df = pd.read_csv(dataset_path)
  print(f"Dataset loaded successfully with shape: {df.shape}")

  # 1. Separate target label (Disease) and raw symptom columns
  if "Disease" in df.columns:
    y = df["Disease"]
    X_raw = df.drop(columns=["Disease"])
  elif "prognosis" in df.columns:
    y = df["prognosis"]
    X_raw = df.drop(columns=["prognosis"])
  else:
    y = df.iloc[:, -1]
    X_raw = df.iloc[:, :-1]

  print("Converting text symptoms into binary feature matrix...")

  # 2. Extract all unique symptoms across the dataset
  unique_symptoms = pd.unique(X_raw.values.ravel())
  unique_symptoms = [
      s
      for s in unique_symptoms
      if pd.notna(s) and str(s).strip() != "" and str(s).lower() != "nan"
  ]

  # 3. Create a binary dataframe (0 or 1 for each unique symptom)
  X_binary = pd.DataFrame(0, index=df.index, columns=unique_symptoms)
  for col in X_raw.columns:
    for symptom in unique_symptoms:
      X_binary[symptom] |= (X_raw[col] == symptom).astype(int)

  X = X_binary
  print(
      f"Feature matrix successfully built with {X.shape[1]} unique symptom"
      " features."
  )

  # 4. Split data into training and testing sets (80% train, 20% test)
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )

  print("Training Random Forest Classifier model...")
  model = RandomForestClassifier(n_estimators=100, random_state=42)
  model.fit(X_train, y_train)

  # 5. Evaluate model accuracy
  y_pred = model.predict(X_test)
  acc = accuracy_score(y_test, y_pred)
  print(f"\nModel Training Complete! Accuracy: {acc * 100:.2f}%")

  # 6. Save the trained model and feature column names
  os.makedirs("models", exist_ok=True)
  joblib.dump(model, "models/model.pkl")
  joblib.dump(list(X.columns), "models/feature_columns.pkl")

  print(
      "SUCCESS: Model saved to 'models/model.pkl' and feature columns saved to"
      " 'models/feature_columns.pkl'!"
  )