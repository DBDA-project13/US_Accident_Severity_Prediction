import os
import joblib
import json
import time
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ---------------- CONFIG ----------------
DATA_DIR = "data/training_ready/"
MODELS_ROOT = "models/final_comparison/"
RANDOM_STATE = 42

MODELS_CONFIG = {
    "xgboost_tuned": {
        "class": XGBClassifier,
        "params": {
            "objective": "multi:softprob",
            "num_class": 4,
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "max_depth": 10,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",  # Optimization for large datasets
            "device": "cpu",        # Change to "cuda" if using GPU
            "random_state": RANDOM_STATE,
            "n_jobs": -1
        }
    },
    "catboost_tuned": {
        "class": CatBoostClassifier,
        "params": {
            "iterations": 1000,
            "learning_rate": 0.05,
            "depth": 8,
            "loss_function": "MultiClass",
            "eval_metric": "TotalF1",
            "task_type": "CPU",     # Change to "GPU" if available
            "verbose": 100,
            "random_seed": RANDOM_STATE,
            "allow_writing_files": False
        }
    }
}

# ---------------- HELPERS ----------------

def load_data():
    print("Loading data...")
    X_train, y_train = joblib.load(os.path.join(DATA_DIR, "train_data.pkl"))
    X_test, y_test = joblib.load(os.path.join(DATA_DIR, "test_data.pkl"))
    # Ensure targets are 0-indexed [0, 1, 2, 3]
    if y_train.min() == 1:
        y_train = y_train - 1
        y_test = y_test - 1
    return X_train, X_test, y_train, y_test

def get_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return weights

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    preds = model.predict(X_test)
    # CatBoost predict often returns a 2D array [[res]], flatten it.
    if len(preds.shape) > 1:
        preds = preds.flatten()
        
    inference_time = time.time() - start_time
    report = classification_report(y_test, preds, output_dict=True)
    
    return {
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "inference_time_cpu": inference_time,
        "class_4_recall": report["3"]["recall"], 
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

def save_all(name, model, metrics, weights):
    path = os.path.join(MODELS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    joblib.dump(model, os.path.join(path, "model.pkl"))
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

# ---------------- MAIN ----------------

def main():
    X_train, X_test, y_train, y_test = load_data()
    weights = get_weights(y_train)
    weights_dict = {i: w for i, w in enumerate(weights)}
    
    comparison_log = []

    for name, config in MODELS_CONFIG.items():
        print(f"\n>>> Starting: {name}")
        params = config["params"].copy()
        model_class = config["class"]
        
        # Training Logic per Model Type
        start_train = time.time()
        
        if name == "xgboost_tuned":
            # XGBoost uses sample_weight instead of class_weight
            sample_weights = np.array([weights_dict[t] for t in y_train])
            model = model_class(**params)
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
        elif name == "catboost_tuned":
            # CatBoost uses class_weights parameter
            params["class_weights"] = weights.tolist()
            model = model_class(**params)
            model.fit(X_train, y_train)
            
        train_time = time.time() - start_train
        
        # Evaluation
        metrics = evaluate_model(model, X_test, y_test)
        metrics["training_time"] = train_time
        
        # Save results
        save_all(name, model, metrics, weights_dict)
        
        comparison_log.append({
            "model": name,
            "macro_f1": metrics["macro_f1"],
            "class_4_recall": metrics["class_4_recall"],
            "train_time": train_time
        })
        
        print(f"Finished {name}. Macro F1: {metrics['macro_f1']:.4f} | C4 Recall: {metrics['class_4_recall']:.4f}")

    print("\n" + "="*40)
    print("FINAL CHALLENGER COMPARISON")
    print("="*40)
    print(pd.DataFrame(comparison_log).sort_values("macro_f1", ascending=False))

if __name__ == "__main__":
    main()