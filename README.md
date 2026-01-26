# US Accident Severity Classification

Project scaffold for building a multi-class classifier to predict accident severity (1..4) from the `us_accident` dataset (~7.7M rows, 46 columns).

Phases:
- Phase 0 - Data Collection and EDA
- Phase 1 — Data preprocessing & feature engineering

- Phase 2 — Model training & evaluation
- Phase 3 — UI with Streamlit
- Phase 4 — Deployment (e.g. AWS)

Structure
```
us_accident_severity/
├─ app/
│  └─ streamlit_app.py
├─ preprocessing/
│  ├─ __init__.py
│  └─ cleaning.py
│  └─ feature.py
├─ src/
│  ├─ __init__.py
│  └─ data_loader.py
├─ data/
│  └─ raw/
│      └─ US_Accidents_March23.csv
│  └─ processed/
│      └─ US_Accidents_Cleaned.parquet
├─ notebooks/
│  └─ EDA.ipynb
├─ tests/
│  └─ test_data_loader.py
│  └─ test.py
├─ scripts/
│  └─ run_local.sh
├─ requirements.txt
└─ .gitignore
```

## Model Comparison Summary

| Model                | Macro F1 | Weighted F1 | Class-4 Recall | Train Time (s) | Inference Time (CPU) |
|---------------------|----------|-------------|----------------|---------------|---------------------|
| Logistic Regression | 0.312    | 0.501       | 0.801          | 3918          | **0.11**            |
| Random Forest       | 0.334    | 0.525       | **0.848**      | 1579          | 6.00                |
| LightGBM (baseline) | 0.409    | 0.654       | 0.821          | 398           | 70.25               |
| LightGBM (tuned)    | 0.454    | 0.710       | 0.820          | 1153          | 435.34              |
| CatBoost (tuned)    | 0.401    | 0.641       | 0.824          | 3774          | **2.70**            |
| **XGBoost (tuned)** | **0.489** | **0.746**   | 0.791          | 1402          | 114.38              |

**Dataset:** ~6.8M rows, highly imbalanced multiclass target (Severity 1–4)

## Final Model Selection: XGBoost

XGBoost was selected as the final production model based on the following criteria:

### 1. Best Overall Predictive Performance
- Achieved the **highest Macro F1 score (0.489)**, indicating superior balanced performance across all severity classes.
- Achieved the **highest Weighted F1 score (0.746)**, reflecting strong real-world accuracy on dominant classes without fully sacrificing minority classes.

### 2. Strong Minority Class Handling
- Severity-4 recall of **~0.79**, which is comparable to LightGBM and CatBoost.
- Significantly better **precision for Severity-4** compared to LightGBM, reducing false severe-accident alarms.

### 3. Robustness on Large-Scale, Imbalanced Data
- XGBoost demonstrated better optimization behavior on:
  - Highly imbalanced class distributions
  - Large sparse feature space after one-hot encoding
- Handled class weights and regularization more effectively than Random Forest and Logistic Regression.

### 4. Acceptable Training & Inference Trade-off
- Training time (~1400s) is acceptable for offline training.
- Inference time (~114ms on CPU) is suitable for batch or near-real-time prediction scenarios.
- LightGBM inference became prohibitively slow after aggressive tuning, whereas XGBoost remained stable.

### 5. Production Stability
- Fewer training warnings compared to LightGBM (e.g., no excessive "no positive gain" splits).
- Better generalization with fewer regularization hacks required.

### Final Decision
Given the balance between **performance, stability, and scalability**, **XGBoost (fine-tuned)** was chosen as the final model for deployment.
