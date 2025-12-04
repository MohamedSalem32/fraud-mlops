# scripts/compute_production_metrics.py

from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)

DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
LOG_PATH = Path("logs/predictions.csv")


def main():
    print("Loading full reference dataset...")
    df_ref = pd.read_csv(DATA_PATH)

    print("Loading logged predictions...")
    df_logs = pd.read_csv(LOG_PATH)

    # Columns to match a logged row with its original dataset row
    join_keys = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
    ]

    # Keep only the needed columns from reference (features + true label)
    df_ref_small = df_ref[["isFraud"] + join_keys].copy()

    print("Merging logs with ground-truth labels...")
    df_join = df_logs.merge(df_ref_small, on=join_keys, how="inner")

    print(f"Joined rows: {len(df_join)} / {len(df_logs)} logged predictions")

    if df_join.empty:
        print("No matches between logs and reference data. "
              "Check join keys or that the stream used the same dataset.")
        return

    y_true = df_join["isFraud"].values
    y_pred = df_join["is_fraud"].values
    y_proba = df_join["fraud_probability"].values

    print("\n=== Classification report on streamed data ===")
    print(classification_report(y_true, y_pred, digits=4))

    # Global ROC-AUC
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC (streamed): {roc_auc:.4f}")

    # PR-AUC (Average Precision)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    print(f"PR-AUC (streamed): {pr_auc:.4f}")

    # Optional: simple view by step range
    df_join["step_bucket"] = pd.cut(
        df_join["step"],
        bins=[0, 100, 200, 300, 400, 500, 600, 750],
        include_lowest=True,
    )

    print("\n=== Fraud rate & predicted probability by step bucket ===")
    bucket_stats = (
        df_join.groupby("step_bucket")
        .agg(
            n=("isFraud", "size"),
            fraud_rate=("isFraud", "mean"),
            mean_pred_proba=("fraud_probability", "mean"),
        )
        .reset_index()
    )
    print(bucket_stats.to_string(index=False))


if __name__ == "__main__":
    main()
