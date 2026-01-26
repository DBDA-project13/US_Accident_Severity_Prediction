import os
import json
import joblib
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score
)
from sklearn.utils.class_weight import compute_class_weight

# ---------------- CONFIG ----------------
MODEL_DIR = "models/lightgbm/"
DATA_DIR = "data/training_ready/"
PREPROCESSOR_PATH = "models/preprocessor.pkl"
RANDOM_STATE = 42
# ---------------------------------------

def load_data():
    X_train, y_train = joblib.load(os.path.join(DATA_DIR, "train_data.pkl"))
    X_test, y_test = joblib.load(os.path.join(DATA_DIR, "test_data.pkl"))
    return X_train, X_test, y_train, y_test

def compute_weights(y_train):
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train
    )
    return dict(zip(classes.tolist(), weights.tolist()))

def train_lightgbm(X_train, y_train, class_weights):
    model = LGBMClassifier(
        objective="multiclass",
        num_class=4,
        class_weight=class_weights,
        n_estimators=300,
        learning_rate=0.1,
        max_depth=-1,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    return model

def evaluate(model, X_test, y_test):
    preds = model.predict(X_test)

    metrics = {
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "weighted_f1": f1_score(y_test, preds, average="weighted"),
        "classification_report": classification_report(y_test, preds, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist()
    }
    return metrics

def save_artifacts(model, metrics, class_weights):
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    with open(os.path.join(MODEL_DIR, "class_weights.json"), "w") as f:
        json.dump(class_weights, f, indent=4)

    # Save feature map
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    feature_names = preprocessor.get_feature_names_out()
    feature_map = {str(i): name for i, name in enumerate(feature_names)}

    with open(os.path.join(MODEL_DIR, "feature_map.json"), "w") as f:
        json.dump(feature_map, f, indent=4)

def main():
    print("Loading training data...")
    X_train, X_test, y_train, y_test = load_data()

    print("Computing class weights...")
    class_weights = compute_weights(y_train)

    print("Training LightGBM...")
    model = train_lightgbm(X_train, y_train, class_weights)

    print("Evaluating model...")
    metrics = evaluate(model, X_test, y_test)

    print("Saving artifacts...")
    save_artifacts(model, metrics, class_weights)

    print("Training complete.")

if __name__ == "__main__":
    main()
