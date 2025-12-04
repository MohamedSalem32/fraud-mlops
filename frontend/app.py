import os
import requests
from datetime import datetime

import pandas as pd
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"
LOG_PATH = os.path.join("logs", "predictions.csv")

st.set_page_config(page_title="Fraud Detection – Dashboard", layout="wide")

st.title("🕵️‍♂️ Real-Time Fraud Detection Dashboard")

tab_test, tab_monitor = st.tabs(["🔍 Manual Testing", "📈 Monitoring"])


# ============ TAB 1: MANUAL TESTING ============
with tab_test:
    st.subheader("Test a Single Transaction")

    st.markdown(
        "Fill in the transaction fields and send them to the FastAPI `/predict` "
        "endpoint. The response will show the fraud probability and decision."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        step = st.number_input("Step (time index)", min_value=1, max_value=743, value=1)
        tx_type = st.selectbox(
            "Type",
            ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"],
            index=1,  # default TRANSFER
        )
        amount = st.number_input("Amount", min_value=0.0, value=1000.0)

    with col2:
        name_orig = st.text_input("Origin account (nameOrig)", value="C123456789")
        oldbalance_org = st.number_input(
            "Old balance origin", min_value=0.0, value=5000.0
        )
        newbalance_org = st.number_input(
            "New balance origin", min_value=0.0, value=4000.0
        )

    with col3:
        name_dest = st.text_input("Destination account (nameDest)", value="C987654321")
        oldbalance_dest = st.number_input(
            "Old balance dest", min_value=0.0, value=0.0
        )
        newbalance_dest = st.number_input(
            "New balance dest", min_value=0.0, value=1000.0
        )

    if st.button("🚀 Send to API"):
        payload = {
            "step": int(step),
            "type": tx_type,
            "amount": float(amount),
            "nameOrig": name_orig,
            "oldbalanceOrg": float(oldbalance_org),
            "newbalanceOrig": float(newbalance_org),
            "nameDest": name_dest,
            "oldbalanceDest": float(oldbalance_dest),
            "newbalanceDest": float(newbalance_dest),
        }

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            st.error(f"Error calling API: {e}")
        else:
            st.success("Prediction received from API ✅")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    label="Fraud probability",
                    value=f"{data['fraud_probability']:.3f}",
                )
                st.metric(
                    label="Is fraud?",
                    value="🚨 FRAUD" if data["is_fraud"] else "✅ Legit",
                )

            with col_b:
                st.write("**Real-time features (from API):**")
                st.json(
                    {
                        "txn_count_24h": data["txn_count_24h"],
                        "avg_amount_24h": data["avg_amount_24h"],
                        "time_since_last_txn_hours": data["time_since_last_txn_hours"],
                        "is_weekend": data["is_weekend"],
                        "amount_zscore": data["amount_zscore"],
                    }
                )


# ============ TAB 2: MONITORING ============
with tab_monitor:
    st.subheader("Monitoring Logged Predictions")

    if not os.path.exists(LOG_PATH):
        st.warning(
            "No logs found yet. Run the stream simulator first "
            "(`python scripts/stream_simulator.py`)."
        )
    else:
        df = pd.read_csv(LOG_PATH)

        st.markdown("### Overview")
        st.write(f"Total logged predictions: **{len(df)}**")

        # Convert timestamp if present
        if "timestamp" in df.columns:
            try:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            except Exception:
                pass

        # Basic fraud stats
        if "is_fraud" in df.columns:
            fraud_rate = df["is_fraud"].mean()
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Fraud rate (logged)", f"{fraud_rate*100:.2f}%")
            with col2:
                st.metric("Num frauds", int(df["is_fraud"].sum()))
            with col3:
                st.metric("Num legit", int((1 - df["is_fraud"]).sum()))

        st.markdown("### Fraud probability distribution")
        if "fraud_probability" in df.columns:
            st.bar_chart(df["fraud_probability"])

        st.markdown("### Sample of recent predictions")
        if "timestamp" in df.columns:
            df_sorted = df.sort_values("timestamp", ascending=False)
        else:
            df_sorted = df.copy()

        st.dataframe(df_sorted.head(20))

        st.info(
            "For deeper analysis (data drift, metrics over time), "
            "you can open the Evidently report at "
            "`reports/data_drift_report.html`."
        )
