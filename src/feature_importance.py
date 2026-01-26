import joblib
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- PATHS ---
MODEL_PATH = "model/fine/lightgbm_tuned/model.pkl" # Adjust if you renamed it
PREPROCESSOR_PATH = "models/preprocessor.pkl"
OUTPUT_DIR = "models/lightgbm_tuned/"

def generate_feature_importance():
    # 1. Load Artifacts
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    
    # 2. Get Feature Names from Preprocessor
    # This handles the OneHotEncoded column names (e.g., 'State_CA', 'State_NY')
    feature_names = preprocessor.get_feature_names_out()
    
    # 3. Get Importance (Gain)
    importances = model.feature_importances_
    
    # 4. Create DataFrame
    fi_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # 5. Save as JSON Feature Map
    # Map index to feature name
    feature_map = {str(i): name for i, name in enumerate(feature_names)}
    with open(os.path.join(OUTPUT_DIR, "feature_map.json"), "w") as f:
        json.dump(feature_map, f, indent=4)
    
    # 6. Save as CSV for easy reading
    fi_df.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
    
    # 7. Visualize Top 20
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=fi_df.head(20))
    plt.title('Top 20 Features by Gain (LightGBM Tuned)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"))
    
    print(f"Feature map and importance plot saved to {OUTPUT_DIR}")
    print("\nTop 15 Most Important Features:")
    print(fi_df.head(15))
    print("\nTop 15 least Important Features:")
    print(fi_df.tail(15))

if __name__ == "__main__":
    generate_feature_importance()