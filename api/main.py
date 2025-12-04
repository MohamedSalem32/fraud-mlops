from fastapi import FastAPI
from pydantic import BaseModel
from typing import Literal, Optional
from datetime import datetime
import joblib
import pandas as pd
import os
import csv
import math

# --------- Config ---------
MODEL_PATH = os.path.join("models", "fraud_detection_pipeline.joblib")
LOG_PATH = os.path.join("logs", "predictions.csv")

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Serves a Random Forest fraud detection model trained in Phase 1.",
    version="0.1.0",
)

# Load model at startup
model = joblib.load(MODEL_PATH)

# In-memory "feature store" for recent user transactions
# key: user_id (nameOrig), value: list of dicts with {step, amount}
user_history = {}


# --------- Request / Response Schemas ---------
class TransactionRequest(BaseModel):
    step: int
    type: Literal["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float


class PredictionResponse(BaseModel):
    fraud_probability: float
    is_fraud: bool
    model: str
    txn_count_24h: int
    avg_amount_24h: float
    time_since_last_txn_hours: Optional[float]
    is_weekend: bool
    amount_zscore: Optional[float]


# --------- Real-time feature engineering ---------
def get_recent_transactions(user_id: str, current_step: int, window_hours: int = 24):
    """Return list of recent transactions for this user in the last `window_hours`."""
    if user_id not in user_history:
        return []
    min_step = current_step - window_hours
    return [t for t in user_history[user_id] if t["step"] >= min_step]


def compute_realtime_features(tx: TransactionRequest):
    """Compute simple sliding-window features based on in-memory history."""
    user_id = tx.nameOrig
    current_step = tx.step
    recent_txns = get_recent_transactions(user_id, current_step, window_hours=24)

    txn_count_24h = len(recent_txns)
    avg_amount_24h = float(
        sum(t["amount"] for t in recent_txns) / txn_count_24h
    ) if txn_count_24h > 0 else 0.0

    # Time since last transaction in "hours" (using step as hour index)
    if recent_txns:
        last_step = max(t["step"] for t in recent_txns)
        time_since_last_txn_hours = float(current_step - last_step)
    else:
        time_since_last_txn_hours = None

    # is_weekend based on real current date
    is_weekend = datetime.utcnow().weekday() >= 5

    # Amount z-score relative to recent history
    if txn_count_24h > 1:
        amounts = [t["amount"] for t in recent_txns]
        mean_amt = sum(amounts) / len(amounts)
        var_amt = sum((a - mean_amt) ** 2 for a in amounts) / len(amounts)
        std_amt = math.sqrt(var_amt)
        if std_amt > 0:
            amount_zscore = (tx.amount - mean_amt) / std_amt
        else:
            amount_zscore = 0.0
    else:
        amount_zscore = None

    return {
        "txn_count_24h": txn_count_24h,
        "avg_amount_24h": avg_amount_24h,
        "time_since_last_txn_hours": time_since_last_txn_hours,
        "is_weekend": is_weekend,
        "amount_zscore": amount_zscore,
    }


def update_history(tx: TransactionRequest):
    """Store current transaction in memory for future feature computation."""
    user_id = tx.nameOrig
    if user_id not in user_history:
        user_history[user_id] = []
    user_history[user_id].append(
        {
            "step": tx.step,
            "amount": tx.amount,
        }
    )


def log_prediction(tx: TransactionRequest, proba: float, is_fraud: bool, rt_feats: dict):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    file_exists = os.path.isfile(LOG_PATH)

    with open(LOG_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "nameOrig",
                    "step",
                    "type",
                    "amount",
                    "oldbalanceOrg",
                    "newbalanceOrig",
                    "nameDest",
                    "oldbalanceDest",
                    "newbalanceDest",
                    "fraud_probability",
                    "is_fraud",
                    "txn_count_24h",
                    "avg_amount_24h",
                    "time_since_last_txn_hours",
                    "is_weekend",
                    "amount_zscore",
                ]
            )

        writer.writerow(
            [
                datetime.utcnow().isoformat(),
                tx.nameOrig,
                tx.step,
                tx.type,
                tx.amount,
                tx.oldbalanceOrg,
                tx.newbalanceOrig,
                tx.nameDest,
                tx.oldbalanceDest,
                tx.newbalanceDest,
                proba,
                int(is_fraud),
                rt_feats["txn_count_24h"],
                rt_feats["avg_amount_24h"],
                rt_feats["time_since_last_txn_hours"]
                if rt_feats["time_since_last_txn_hours"] is not None
                else "",
                rt_feats["is_weekend"],
                rt_feats["amount_zscore"]
                if rt_feats["amount_zscore"] is not None
                else "",
            ]
        )


# --------- Routes ---------
@app.get("/")
def root():
    return {"status": "ok", "message": "Fraud detection API is running."}


@app.post("/predict", response_model=PredictionResponse)
def predict(tx: TransactionRequest):
    # 1) Prepare input for the model (same features as in Phase 1)
    input_df = pd.DataFrame(
        [
            {
                "step": tx.step,
                "type": tx.type,
                "amount": tx.amount,
                "oldbalanceOrg": tx.oldbalanceOrg,
                "newbalanceOrig": tx.newbalanceOrig,
                "oldbalanceDest": tx.oldbalanceDest,
                "newbalanceDest": tx.newbalanceDest,
            }
        ]
    )

    # 2) Model prediction
    proba = float(model.predict_proba(input_df)[0, 1])
    is_fraud = proba >= 0.5  # simple threshold; can be tuned

    # 3) Real-time style features
    rt_feats = compute_realtime_features(tx)

    # 4) Update history and log
    update_history(tx)
    log_prediction(tx, proba, is_fraud, rt_feats)

    # 5) Build response
    return PredictionResponse(
        fraud_probability=proba,
        is_fraud=is_fraud,
        model="RandomForest_v1",
        txn_count_24h=rt_feats["txn_count_24h"],
        avg_amount_24h=rt_feats["avg_amount_24h"],
        time_since_last_txn_hours=rt_feats["time_since_last_txn_hours"],
        is_weekend=rt_feats["is_weekend"],
        amount_zscore=rt_feats["amount_zscore"],
    )
