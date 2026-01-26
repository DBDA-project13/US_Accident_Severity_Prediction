import os
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import time

# --- CONFIGURATION ---
PROCESSED_DATA_DIR = "data/training_ready/"
MODELS_DIR = "models/"

def load_processed_data():
    """Loads the sparse matrices saved by the pipeline."""
    print("Loading processed data...")
    t0 = time.time()
    
    # Load Train
    X_train, y_train = joblib.load(os.path.join(PROCESSED_DATA_DIR, "train_data.pkl"))
    
    # Load Test
    X_test, y_test = joblib.load(os.path.join(PROCESSED_DATA_DIR, "test_data.pkl"))
    
    print(f"Data loaded in {time.time() - t0:.2f}s")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y_train, X_test, y_test

def create_folder_structure(model_name):
    """Creates the folder structure: models/model_name/"""
    path = os.path.join(MODELS_DIR, model_name)
    os.makedirs(path, exist_ok=True)
    return path

def save_artifacts(model, metrics, path):
    """Saves model.pkl and metrics.json"""
    # 1. Save Model
    print(f"Saving model to {path}...")
    joblib.dump(model, os.path.join(path, "model.pkl"))
    
    # 2. Save Metrics
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    print("Artifacts saved.")

def train_and_evaluate(model_name, model, X_train, y_train, X_test, y_test):
    print(f"\n{'='*10} Training {model_name} {'='*10}")
    output_path = create_folder_structure(model_name)
    
    # 1. Train
    t0 = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - t0
    print(f"Training completed in {training_time:.2f}s")
    
    # 2. Predict
    print("Evaluating...")
    y_pred = model.predict(X_test)
    
    # 3. Calculate Metrics
    # Weighted F1 is better for imbalanced data than Accuracy
    f1 = f1_score(y_test, y_pred, average='weighted')
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Extract key metrics for simple JSON
    metrics = {
        "model_name": model_name,
        "training_time_seconds": round(training_time, 2),
        "f1_weighted": round(f1, 4),
        "accuracy": round(report["accuracy"], 4),
        "macro_avg_f1": round(report["macro avg"]["f1-score"], 4),
        # Class-specific performance (crucial for severity)
        "class_performance": {
            "1": report.get("1", {}),
            "2": report.get("2", {}),
            "3": report.get("3", {}),
            "4": report.get("4", {})
        }
    }
    
    print(f"Performance (Weighted F1): {f1:.4f}")
    
    # 4. Save
    save_artifacts(model, metrics, output_path)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_processed_data()
    
    # --- MODEL ZOO ---
    # We define the models here. 
    # 'class_weight="balanced"' is the magic sauce for your dataset.
    
    models_to_train = {
        "logistic_regression": LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            n_jobs=-1,  # Use all cores
            solver='saga' # Saga is faster for large datasets
        ),
        
        # WARNING: RF on 7.5M rows is slow. 
        # I limited n_estimators to 50 and max_depth to 20 for this first run.
        # Increase these for production performance later.
        "random_forest": RandomForestClassifier(
            class_weight='balanced',
            n_estimators=50, 
            max_depth=20,
            n_jobs=-1,
            random_state=42
        )
    }
    
    for name, model_obj in models_to_train.items():
        train_and_evaluate(name, model_obj, X_train, y_train, X_test, y_test)