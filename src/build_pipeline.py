import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- CONFIGURATION ---
DATA_PATH = "data/processed/US_Accidents_Cleaned.parquet"
ARTIFACTS_DIR = "models/"
PROCESSED_DATA_DIR = "data/training_ready/" # Where we save ready-to-train data

# Columns definitions
NUMERIC_FEATURES = [
    "Temperature(F)", "Humidity(%)", "Pressure(in)", "Visibility(mi)", 
    "Wind_Speed(mph)", "Start_Lat", "Start_Lng", "hour", "day", "month", 
    "Distance(mi)_capped", "Distance(mi)_log"
]

# We treat booleans as numeric (0/1) or categorical? 
# Usually Scikit-learn treats bools as numeric 0/1 automatically, 
# but OHE is safer if you want explicit handling. Let's keep them as pass-through (already 0/1).
BOOL_FEATURES = [
    "Amenity", "Bump", "Crossing", "Give_Way", "Junction", "No_Exit", 
    "Railway", "Roundabout", "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
]

CATEGORICAL_FEATURES = [
    "State", "Weather_Simple", "Wind_Direction_Simple", "time_bucket", "Distance(mi)_bin"
]

TARGET = "Severity"

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_parquet(DATA_PATH)
    return df

def get_preprocessor():
    """
    Creates a scikit-learn ColumnTransformer.
    This is the object we will save for the UI later.
    """
    # Pipeline for numerical features: Scale them
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline for categorical features: One Hot Encode
    # handle_unknown='ignore' is CRITICAL for production (if a new category appears in future)
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True)) 
    ])

    # Combine them
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bool', 'passthrough', BOOL_FEATURES) # Don't touch booleans
        ],
        remainder='drop' # Drop any columns not listed (sanity check)
    )
    
    return preprocessor

def run_pipeline(sample_fraction=1.0):
    """
    sample_fraction: Set to 0.1 to run on 10% of data for debugging/quick tests
    """
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    df = load_data()

    # --- 1. SAMPLING (Optional for Dev) ---
    # if sample_fraction < 1.0:
    #     print(f"Sampling {sample_fraction*100}% of data for rapid iteration...")
    #     df = df.sample(frac=sample_fraction, random_state=42)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # --- 2. STRATIFIED SPLIT ---
    # We MUST split before scaling to avoid data leakage.
    # Stratify is crucial because Class 1 is only 0.8%
    print("Splitting data (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # --- 3. FIT PREPROCESSOR ---
    print("Fitting preprocessor on TRAIN data...")
    preprocessor = get_preprocessor()
    
    # Fit on train, transform train
    # Note: This returns a Sparse Matrix (memory efficient) because of OHE
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform test (DO NOT FIT)
    X_test_processed = preprocessor.transform(X_test)

    print(f"Processed Train Shape: {X_train_processed.shape}")
    print(f"Processed Test Shape: {X_test_processed.shape}")

    # --- 4. SAVE ARTIFACTS ---
    print("Saving preprocessor object...")
    joblib.dump(preprocessor, os.path.join(ARTIFACTS_DIR, "preprocessor.pkl"))

    # Saving processed matrices (Optional: Only if you have disk space and want to skip processing next time)
    # Since these are sparse matrices, use scipy's save_npz or joblib
    print("Saving processed datasets...")
    joblib.dump((X_train_processed, y_train), os.path.join(PROCESSED_DATA_DIR, "train_data.pkl"))
    joblib.dump((X_test_processed, y_test), os.path.join(PROCESSED_DATA_DIR, "test_data.pkl"))

    print("Pipeline Complete. Ready for Model Training.")

if __name__ == "__main__":
    # Set sample_fraction=1.0 for full run, or 0.1 for testing
    run_pipeline(sample_fraction=1.0)