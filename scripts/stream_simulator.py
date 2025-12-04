import os
import time
import requests
import pandas as pd

API_URL = "http://127.0.0.1:8000/predict"
DATA_PATH = os.path.join("data", "PS_20174392719_1491204439457_log.csv")

# Number of transactions to send (to avoid spamming)
N_TXNS = 200  # change to 1000, etc. if you want


def row_to_payload(row):
    """Map a dataframe row to the JSON schema expected by the API."""
    return {
        "step": int(row["step"]),
        "type": str(row["type"]),
        "amount": float(row["amount"]),
        "nameOrig": str(row["nameOrig"]),
        "oldbalanceOrg": float(row["oldbalanceOrg"]),
        "newbalanceOrig": float(row["newbalanceOrig"]),
        "nameDest": str(row["nameDest"]),
        "oldbalanceDest": float(row["oldbalanceDest"]),
        "newbalanceDest": float(row["newbalanceDest"]),
    }


def main():
    print("Loading data from", DATA_PATH)
    df = pd.read_csv(DATA_PATH)

    # For realism, you can filter to a subset of types, e.g. TRANSFER + CASH_OUT
    df_stream = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].head(N_TXNS)

    print(f"Streaming {len(df_stream)} transactions to {API_URL} ...")

    for i, (_, row) in enumerate(df_stream.iterrows(), start=1):
        payload = row_to_payload(row)

        try:
            resp = requests.post(API_URL, json=payload, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            print(f"[{i}] Error calling API:", e)
            continue

        # Print a compact summary every 20 transactions
        if i % 20 == 0 or i == 1:
            print(
                f"[{i}] step={payload['step']}, user={payload['nameOrig']}, "
                f"amount={payload['amount']:.2f} -> "
                f"proba={data['fraud_probability']:.3f}, is_fraud={data['is_fraud']}"
            )

        # Small delay to simulate time between transactions
        time.sleep(0.1)

    print("Done streaming.")
    print("Check logs/predictions.csv for the logged predictions and features.")


if __name__ == "__main__":
    main()
