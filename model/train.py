
import os
import joblib
import json
import time
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

# ---------------- CONFIG ----------------
DATA_DIR = "data/training_ready/"
MODELS_ROOT = "model/fine/"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
RANDOM_STATE = 42
MODELS_CONFIG ={"lightgbm_tuned": {
    "class": LGBMClassifier,
    "params": {
        "objective": "multiclass",
        "num_class": 4,
        "n_estimators": 1000,        # Increased from 500
        "learning_rate": 0.03,      # Lowered for better convergence
        "num_leaves": 150,          # Key for complex patterns
        "max_depth": 12,            # Prevents deep overfit
        "min_child_samples": 200,   # High value for large datasets
        "colsample_bytree": 0.8,    # Feature fraction
        "subsample": 0.8,           # Data fraction
        "reg_alpha": 0.1,           # L1 Regularization
        "reg_lambda": 0.1,          # L2 Regularization
        "importance_type": "gain",
        "n_jobs": -1,
        "random_state": RANDOM_STATE
    }
}}

# ---------------- HELPERS ----------------

def load_data():
    X_train, y_train = joblib.load(os.path.join(DATA_DIR, "train_data.pkl"))
    X_test, y_test = joblib.load(os.path.join(DATA_DIR, "test_data.pkl"))
    # XGBoost/LightGBM expect 0-indexed targets: [1,2,3,4] -> [0,1,2,3]
    return X_train, X_test, y_train - 1, y_test - 1

def get_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    return dict(zip(classes, weights))

def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    preds = model.predict(X_test)
    inference_time = time.time() - start_time
    
    report = classification_report(y_test, preds, output_dict=True)
    
    return {
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "inference_time_cpu": inference_time,
        "class_4_recall": report["3"]["recall"], # index 3 is Severity 4
        "report": report,
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }

def save_all(name, model, metrics, weights):
    path = os.path.join(MODELS_ROOT, name)
    os.makedirs(path, exist_ok=True)
    
    joblib.dump(model, os.path.join(path, "model.pkl"))
    
    with open(os.path.join(path, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)
        
    with open(os.path.join(path, "class_weights.json"), "w") as f:
        json.dump({int(k)+1: v for k, v in weights.items()}, f, indent=4)

# ---------------- MAIN ----------------

def main():
    X_train, X_test, y_train, y_test = load_data()
    weights_dict = get_weights(y_train)
    
    comparison_log = []

    for name, config in MODELS_CONFIG.items():
        print(f"\n>>> Starting: {name}")
        
        # Inject weights
        params = config["params"].copy()
        if name == "xgboost":
            # XGBoost handles weights per-sample, not per-class in fit
            sample_weights = np.array([weights_dict[t] for t in y_train])
        else:
            params["class_weight"] = weights_dict
        
        model = config["class"](**params)
        
        # Train
        start_train = time.time()
        if name == "xgboost":
            model.fit(X_train, y_train, sample_weight=sample_weights)
        else:
            model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # Eval
        metrics = evaluate_model(model, X_test, y_test)
        metrics["training_time"] = train_time
        
        # Save
        save_all(name, model, metrics, weights_dict)
        
        comparison_log.append({
            "model": name,
            "macro_f1": metrics["macro_f1"],
            "class_4_recall": metrics["class_4_recall"],
            "train_time": train_time
        })
        
        print(f"Finished {name}. Macro F1: {metrics['macro_f1']:.4f} | C4 Recall: {metrics['class_4_recall']:.4f}")

    # Final Comparison Summary
    print("\n" + "="*30)
    print("FINAL MODEL COMPARISON")
    print("="*30)
    print(pd.DataFrame(comparison_log).sort_values("macro_f1", ascending=False))

if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import seaborn as sns

def save_feature_importance(model, feature_names, name):
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
    fi_df = fi_df.sort_values('importance', ascending=False).head(20)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=fi_df)
    plt.title(f'Top 20 Features - {name}')
    plt.savefig(f"models/{name}/feature_importance.png")