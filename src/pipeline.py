"""
Reusable pipeline for the Mobile App User Engagement Prediction project.

The pipeline:
1. Loads the raw event log CSV.
2. Cleans timestamps and critical identifiers.
3. Restricts the analytical period to the valid 2024-2025 window.
4. Creates a fixed prediction cutoff with a complete 7-day future window.
5. Engineers user-level engagement features.
6. Creates the binary target active_next_7_days.
7. Trains Logistic Regression and Random Forest.
8. Saves evaluation outputs and plots.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class Config:
    data_path: Path
    results_dir: Path
    figures_dir: Path
    random_state: int = 42
    test_size: float = 0.20


def load_and_clean_data(path: Path) -> pd.DataFrame:
    """Load raw event logs and perform conservative cleaning."""
    df = pd.read_csv(path)
    required = {"timestamp", "user_id", "session_id"}
    missing_required = required - set(df.columns)
    if missing_required:
        raise ValueError(f"Missing required columns: {sorted(missing_required)}")

    df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # Valid observation window established during exploratory audit.
    df = df[
        df["timestamp_parsed"].notna()
        & df["user_id"].notna()
        & df["session_id"].notna()
        & df["timestamp_parsed"].dt.year.isin([2024, 2025])
    ].copy()

    # Standardize non-critical categorical columns.
    categorical = [
        "device_os",
        "device_os_version",
        "device_model",
        "screen_resolution",
        "location_country",
        "location_city",
    ]
    for col in categorical:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    return df.sort_values(["timestamp_parsed", "user_id"]).reset_index(drop=True)


def build_model_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Timestamp, pd.Timestamp]:
    """Create user-level features and a fixed 7-day future-activity target."""
    data_end = df["timestamp_parsed"].max()
    cutoff = data_end - pd.Timedelta(days=7)

    historical = df[df["timestamp_parsed"] <= cutoff].copy()
    future = df[
        (df["timestamp_parsed"] > cutoff)
        & (df["timestamp_parsed"] <= cutoff + pd.Timedelta(days=7))
    ].copy()

    def mode_or_unknown(series: pd.Series) -> str:
        non_null = series.dropna()
        if non_null.empty:
            return "Unknown"
        return str(non_null.mode().iloc[0])

    user = historical.groupby("user_id").agg(
        total_events=("user_id", "size"),
        unique_sessions=("session_id", "nunique"),
        active_days=("timestamp_parsed", lambda s: s.dt.date.nunique()),
        first_activity=("timestamp_parsed", "min"),
        last_activity=("timestamp_parsed", "max"),
        device_os=("device_os", mode_or_unknown),
        location_country=("location_country", mode_or_unknown),
    ).reset_index()

    user["recency_days"] = (
        cutoff - user["last_activity"]
    ).dt.total_seconds() / 86400.0

    user["events_per_session"] = (
        user["total_events"] / user["unique_sessions"].replace(0, np.nan)
    ).fillna(0)

    future_users = set(future["user_id"].unique())
    user["active_next_7_days"] = user["user_id"].isin(future_users).astype(int)

    cols = [
        "user_id",
        "total_events",
        "unique_sessions",
        "active_days",
        "recency_days",
        "events_per_session",
        "device_os",
        "location_country",
        "active_next_7_days",
    ]
    return user[cols], cutoff, data_end


def build_models(random_state: int = 42):
    """Return configured Logistic Regression and Random Forest pipelines."""
    numeric = [
        "total_events",
        "unique_sessions",
        "active_days",
        "recency_days",
        "events_per_session",
    ]
    categorical = ["device_os", "location_country"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ]
    )

    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(
    model_df: pd.DataFrame, results_dir: Path, figures_dir: Path, random_state: int = 42, test_size: float = 0.20
) -> pd.DataFrame:
    """Train models, save metrics, and generate evaluation figures."""
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = model_df.drop(columns=["active_next_7_days", "user_id"])
    y = model_df["active_next_7_days"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    results = []
    fitted: Dict[str, Pipeline] = {}
    predictions: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for name, pipeline in build_models(random_state).items():
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        fitted[name] = pipeline
        predictions[name] = (pred, proba)

        results.append(
            {
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1-Score": f1_score(y_test, pred, zero_division=0),
                "ROC-AUC": roc_auc_score(y_test, proba),
                "Average Precision": average_precision_score(y_test, proba),
            }
        )

    metrics = pd.DataFrame(results).sort_values(
        ["ROC-AUC", "F1-Score"], ascending=False
    )
    metrics.to_csv(results_dir / "model_comparison.csv", index=False)

    # Confusion matrix for the selected model by ROC-AUC/F1 tie-break.
    best_name = metrics.iloc[0]["Model"]
    best_pred, best_proba = predictions[best_name]
    cm = confusion_matrix(y_test, best_pred)
    pd.DataFrame(
        cm,
        index=["Actual 0", "Actual 1"],
        columns=["Predicted 0", "Predicted 1"],
    ).to_csv(results_dir / "confusion_matrix.csv")

    # ROC
    plt.figure(figsize=(7, 5.5))
    for name, (_, proba) in predictions.items():
        fpr, tpr, _ = roc_curve(y_test, proba)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, proba):.3f})")
    plt.plot([0, 1], [0, 1], "--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Predictive Models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "roc_curves.png", dpi=300)
    plt.close()

    # PR
    plt.figure(figsize=(7, 5.5))
    for name, (_, proba) in predictions.items():
        precision, recall, _ = precision_recall_curve(y_test, proba)
        ap = average_precision_score(y_test, proba)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "precision_recall_curves.png", dpi=300)
    plt.close()

    # Confusion matrix
    plt.figure(figsize=(6.5, 5.5))
    plt.imshow(cm, interpolation="nearest")
    plt.colorbar()
    plt.xticks([0, 1], ["Not Active (0)", "Active (1)"], rotation=20)
    plt.yticks([0, 1], ["Not Active (0)", "Active (1)"])
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(f"Confusion Matrix - {best_name}")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=300)
    plt.close()

    return metrics


def run(config: Config) -> None:
    """Run the complete project pipeline."""
    config.results_dir.mkdir(parents=True, exist_ok=True)
    config.figures_dir.mkdir(parents=True, exist_ok=True)

    raw = load_and_clean_data(config.data_path)
    model_df, cutoff, data_end = build_model_dataset(raw)

    model_df.to_csv(config.results_dir / "model_ready_user_engagement_data.csv", index=False)

    summary = pd.DataFrame(
        [
            ["Valid cleaned events", len(raw)],
            ["Unique users in model-ready data", model_df["user_id"].nunique()],
            ["Prediction cutoff", cutoff],
            ["Observation end", data_end],
            ["Active next 7 days = 1", int(model_df["active_next_7_days"].sum())],
            ["Active next 7 days = 0", int((model_df["active_next_7_days"] == 0).sum())],
        ],
        columns=["Metric", "Value"],
    )
    summary.to_csv(config.results_dir / "pipeline_summary.csv", index=False)

    evaluate_models(
        model_df=model_df,
        results_dir=config.results_dir,
        figures_dir=config.figures_dir,
        random_state=config.random_state,
        test_size=config.test_size,
    )


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    config = Config(
        data_path=ROOT / "data" / "mobile_app_interactions.csv",
        results_dir=ROOT / "results",
        figures_dir=ROOT / "figures" / "model_evaluation",
    )
    run(config)
