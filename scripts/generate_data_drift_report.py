# scripts/generate_data_drift_report.py

import pandas as pd
from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

# Paths
DATA_PATH = Path("data/PS_20174392719_1491204439457_log.csv")
LOG_PATH = Path("logs/predictions.csv")
REPORT_DIR = Path("reports")
REPORT_DIR.mkdir(exist_ok=True)
REPORT_PATH = REPORT_DIR / "data_drift_report.html"


def main():
    print("Loading reference (training-style) data...")
    df_ref = pd.read_csv(DATA_PATH)

    # Keep only the main input features (same logic as in training)
    ref_cols = [
        "step",
        "type",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    df_ref = df_ref[ref_cols].copy()

    print("Loading current (production-style) data from logs...")
    df_cur = pd.read_csv(LOG_PATH)

    # Align columns between reference and current
    # (take only the columns present in both)
    common_cols = list(set(df_ref.columns).intersection(df_cur.columns))
    df_ref = df_ref[common_cols].copy()
    df_cur = df_cur[common_cols].copy()

    print("Building Evidently data drift report...")
    report = Report([
        DataDriftPreset()
    ])

    # In 0.7.x API you can pass dataframes directly:
    # report.run(reference_data, current_data)
    result = report.run(df_ref, df_cur)

    # Save HTML report
    result.save_html(str(REPORT_PATH))
    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
