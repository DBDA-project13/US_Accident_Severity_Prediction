# US Accident Severity Classification

Project scaffold for building a multi-class classifier to predict accident severity (1..4) from the `us_accident` dataset (~7.7M rows, 46 columns).

Phases:
- Phase 1 — Data preprocessing & feature engineering (current)
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
├─ src/
│  ├─ __init__.py
│  └─ data_loader.py
├─ data/            # put raw CSV(s) and small samples here (gitignore large files)
├─ notebooks/
│  └─ EDA.ipynb
├─ tests/
│  └─ test_data_loader.py
├─ scripts/
│  └─ run_local.sh
├─ requirements.txt
└─ .gitignore
```

Quickstart

1. Create a Python venv and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. To run the Streamlit app locally (after installing deps):

```bash
bash scripts/run_local.sh
```

Notes
- Keep the raw dataset out of git; place it into `data/` and reference it from `src/data_loader.py`.
- Next step: implement preprocessing functions in `preprocessing/cleaning.py` and add an exploratory notebook in `notebooks/EDA.ipynb`.
