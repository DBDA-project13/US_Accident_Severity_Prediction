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

