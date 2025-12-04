# Real-Time Fraud Detection with MLOps Pipeline

End-to-end project that trains a fraud detection model on transaction data, serves it as a real-time API, simulates streaming traffic, and monitors the system with dashboards and reports.

This project is designed as a **portfolio / PFE-ready** example to demonstrate skills in:

- Machine Learning on imbalanced data
- FastAPI model serving
- Real-time feature engineering
- Logging & monitoring (Evidently, custom scripts, Streamlit)
- (Optional) Containerization with Docker

---

## 1. Business Problem

Online payment fraud causes major financial losses for banks, merchants, and customers.  
The goal is to detect fraudulent transactions early while keeping false alarms under control.

**Objective**

> Train and deploy a model that predicts `isFraud` for each transaction, monitor its behavior on "live" traffic, and expose simple dashboards for testing and monitoring.

Key challenges:

- **Strong class imbalance**: fraud ≈ 0.13% of all transactions.
- Need metrics beyond simple accuracy (precision, recall, PR-AUC).
- Need an architecture that makes sense for an MLOps / production setting.

---

## 2. Dataset

**Dataset**: [Online Payments Fraud Detection (PaySim)](https://www.kaggle.com/datasets/ealaxi/paysim1)  
Main columns used:

- `step`: time step (1 step = 1 hour)
- `type`: transaction type (`PAYMENT`, `TRANSFER`, `CASH_OUT`, `CASH_IN`, `DEBIT`)
- `amount`: transaction amount
- `oldbalanceOrg`, `newbalanceOrig`: origin account balances
- `oldbalanceDest`, `newbalanceDest`: destination account balances
- `isFraud`: target label (1 = fraud, 0 = legit)

> For licensing/size reasons, you can keep only a sampled CSV in `data/` and link to Kaggle for the full dataset.

---

## 3. Architecture Overview

```text
          ┌─────────────────────────┐
          │  Offline Training (Kaggle)
          │  - EDA + preprocessing
          │  - Logistic Regression, RandomForest
          │  - Metrics: precision/recall, ROC-AUC, PR-AUC
          │  - Export best pipeline (RF) with joblib
          └──────────────┬──────────┘
                         │ fraud_detection_pipeline.joblib
                         ▼
               ┌─────────────────────┐
               │ FastAPI Inference   │
               │  - Loads RF pipeline
               │  - Real-time features per user:
               │      • txn_count_24h
               │      • avg_amount_24h
               │      • time_since_last_txn
               │      • is_weekend
               │      • amount_zscore
               │  - Logs predictions to CSV
               └────────────┬────────┘
                            │ /predict
                            ▼
                ┌──────────────────────┐
                │ Stream Simulator      │
                │  - Reads original CSV │
                │  - Sends txns to API  │
                └───────────┬──────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Logs & Monitoring    │
                │  - logs/predictions.csv
                │  - Evidently drift report
                │  - Custom metrics script
                │  - Streamlit dashboard
                └──────────────────────┘
```

---

## 4. Project Structure

```text
fraud-mlops/
├── api/
│   └── main.py                    # FastAPI app + real-time features + logging
├── data/
│   └── PS_2017...log_sample.csv   # Sample of the Kaggle dataset
├── frontend/
│   └── app.py                     # Streamlit dashboard (testing + monitoring)
├── logs/
│   └── predictions.csv            # Logged predictions & features (generated)
├── models/
│   └── fraud_detection_pipeline.joblib  # Trained RandomForest pipeline
├── reports/
│   └── data_drift_report.html     # Evidently data drift report (generated)
├── scripts/
│   ├── stream_simulator.py        # Simulate transaction stream
│   ├── generate_data_drift_report.py  # Build Evidently drift report
│   └── compute_production_metrics.py  # Metrics on logged predictions
├── requirements.txt
├── Dockerfile                     # (Optional) container for the API
└── README.md
```

The full training notebook (EDA + model selection + export) is available on Kaggle:  
👉 [Fraud Detection – Offline Training Notebook](https://www.kaggle.com/code/mohamedmedsalem/frauddetection)

---

## 5. Phase 1 – Offline Model Training (Kaggle)

All offline experimentation was done in the Kaggle notebook linked above.

### 5.1 EDA & Class Imbalance

- Inspected feature distributions and class frequencies; **fraud ≈ 0.13%** of all transactions.
- Explained why **accuracy is misleading** in this context and focused instead on:
  - **Precision, recall, F1** for the fraud class (`isFraud = 1`),
  - **ROC-AUC**,
  - **PR-AUC (Average Precision)**, which is more informative under heavy imbalance.

### 5.2 Preprocessing

- Dropped:
  - `isFlaggedFraud`,
  - `nameOrig`,
  - `nameDest`  
  (the last two are high-cardinality IDs that are not directly useful in simple tabular models).
- Performed a **train/test split with stratification** on `isFraud` to preserve the class ratio.
- Built a `ColumnTransformer`:
  - Numeric features → `StandardScaler`
  - Categorical feature `type` → `OneHotEncoder(handle_unknown="ignore")`

### 5.3 Models

Two main models were trained:

- **Logistic Regression** with `class_weight="balanced"`
- **RandomForestClassifier** with `class_weight="balanced"`

### 5.4 Evaluation & Model Selection

- Compared:
  - Confusion matrices
  - Classification reports (with a focus on class `1`)
  - ROC-AUC
  - PR-AUC
  - Precision–Recall curves

The **Random Forest** achieved much higher **PR-AUC** and a better **precision/recall trade-off for fraud**, so it was chosen as the production model.

> ✅ **Selected model for deployment**: RandomForest pipeline (preprocessing + classifier).

### 5.5 Export

The full preprocessing + model pipeline was exported with `joblib`:

```python
joblib.dump(rf_pipeline, "fraud_detection_pipeline.joblib")
```

This file is stored in `models/fraud_detection_pipeline.joblib` and loaded by the API.

---

## 6. Phase 2 – Real-Time Inference Service (FastAPI)

`api/main.py` implements the FastAPI inference service.

### 6.1 Main Responsibilities

- Load the trained pipeline:

  ```python
  model = joblib.load("models/fraud_detection_pipeline.joblib")
  ```

- Define a Pydantic model `TransactionRequest` containing all transaction fields.
- Maintain an **in-memory history** per `nameOrig` to compute real-time features:
  - `txn_count_24h`
  - `avg_amount_24h`
  - `time_since_last_txn_hours`
  - `is_weekend`
  - `amount_zscore` (current amount relative to recent user history)

- Expose two endpoints:
  - `GET /` – simple health check  
  - `POST /predict` – main inference endpoint

---

### 6.2 `/predict` Response Format

Example JSON response:

```json
{
  "fraud_probability": 0.0269,
  "is_fraud": false,
  "model": "RandomForest_v1",
  "txn_count_24h": 5,
  "avg_amount_24h": 12345.6,
  "time_since_last_txn_hours": 1.0,
  "is_weekend": false,
  "amount_zscore": -0.42
}
```

- `fraud_probability`: model's probability for `isFraud = 1`
- `is_fraud`: boolean decision based on a configurable threshold
- The remaining fields are real-time features used for monitoring and debugging.

### 6.3 Logging

Every call to `/predict` is logged to `logs/predictions.csv`, including:

- Raw input fields
- Model outputs
- Real-time features
- Timestamp

These logs are later used by the monitoring scripts and the Streamlit dashboard.

---

## 7. Phase 2 – Stream Simulation

`scripts/stream_simulator.py` simulates live traffic to the API.

### What it does:

- Reads transactions from the CSV in `data/`.
- Optionally filters only relevant transaction types (e.g. `TRANSFER`, `CASH_OUT`).
- Sends them one by one to the `/predict` endpoint using the `requests` library.
- Prints progress information in the console.
- Populates `logs/predictions.csv` with the responses returned by the API.

### Usage:

```bash
python scripts/stream_simulator.py
```

*Make sure the FastAPI service is running before launching the simulator.*

---

## 8. Phase 3 – Monitoring & Dashboards

### 8.1 Evidently Data Drift

`scripts/generate_data_drift_report.py` builds a data drift report using Evidently.

- **Reference data**: training-style sample from the original CSV.
- **Current data**: logged predictions from `logs/predictions.csv`.

It uses `DataDriftPreset` to compare the two distributions and saves an HTML report:

```bash
python scripts/generate_data_drift_report.py
```

The report is written to:

```text
reports/data_drift_report.html
```

Open this file in a browser to explore drift metrics and visualizations.

---

### 8.2 Production Metrics

`scripts/compute_production_metrics.py` evaluates the model on actual streamed transactions.

#### Main steps:

- Joins `logs/predictions.csv` back to the original dataset using the feature columns.
- Computes performance metrics on this streamed subset:
  - Precision, recall, F1-score
  - ROC-AUC
  - PR-AUC
- Aggregates results by step buckets (time ranges) to report:
  - Fraud rate per bucket
  - Mean predicted fraud probability per bucket

#### Usage:

```bash
python scripts/compute_production_metrics.py
```

---

### 8.3 Streamlit Dashboard

`frontend/app.py` provides a lightweight Streamlit UI with two tabs:

**1. Manual Testing**

- Form to send a single transaction to `/predict`.
- Shows:
  - Fraud probability
  - Final decision (`is_fraud`)
  - All real-time feature values

**2. Monitoring Logged Predictions**

- Reads `logs/predictions.csv`.
- Shows:
  - Total number of predictions
  - Fraud rate
  - Number of fraud / legit predictions
  - Distribution of fraud probabilities (simple chart)
  - Table with the most recent predictions

#### Usage (with the API running):

```bash
streamlit run frontend/app.py
```

---

## 9. How to Run Locally

### Prerequisites

Create and activate a virtual environment (Python 3.10 recommended):

```bash
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

### 9.1 Start FastAPI

```bash
uvicorn api.main:app --reload
```

Interactive docs: http://127.0.0.1:8000/docs

---

### 9.2 Run Stream Simulator (optional)

```bash
python scripts/stream_simulator.py
```

---

### 9.3 Launch Streamlit Dashboard

```bash
streamlit run frontend/app.py
```

---

### 9.4 Generate Monitoring Artifacts

```bash
python scripts/generate_data_drift_report.py
python scripts/compute_production_metrics.py
```

---

## 10. Tech Stack

- **Python**, pandas, NumPy
- **scikit-learn** (RandomForest, pipelines, metrics)
- **FastAPI**, Pydantic, Uvicorn
- **Streamlit** for the dashboard
- **Evidently** for data drift analysis
- **(Optional)** Docker for containerizing the API

---

## 11. .gitignore

Before pushing to GitHub, create a `.gitignore` file so you don't upload huge or unnecessary files:

```gitignore
# Python
__pycache__/
*.py[cod]
*.pyo
*.pyd
*.log

# Virtual envs
.venv/
.venv310/

# OS
.DS_Store
Thumbs.db

# Data / logs
data/PS_20174392719_1491204439457_log.csv
logs/*.csv

# Streamlit cache
.streamlit/

# Jupyter checkpoints
.ipynb_checkpoints/
```

You can keep a small sample CSV in `data/` if you want (e.g. `PS_2017439..._sample.csv`) and do not ignore that one, so the repo still contains a tiny example for local testing.
---
## 12. Contact

For questions or collaboration:

- **GitHub**: [MohamedSalem32](https://github.com/MohamedSalem32)
- **LinkedIn**: [Mohamed Mohamed Salem](https://www.linkedin.com/in/mohamed-mohamed-salem-5331a5256/)
- **Email**: [mohamed.salem@esprit.tn](mailto:mohamed.salem@esprit.tn)