import math
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "catboost_identity_risk_model.cbm"
DATA_PATH = BASE_DIR / "dataset" / "identity_risk_train.csv"
OUTPUT_DIR = BASE_DIR / "outputs" / "model_evaluation"
TARGET = "target_label"
DEFAULT_THRESHOLD = 0.55
CHART_THRESHOLDS = [
    0.01,
    0.11,
    0.21,
    0.31,
    0.41,
    0.50,
    0.51,
    0.55,
    0.60,
    0.61,
    0.70,
    0.71,
    0.80,
    0.81,
    0.90,
    0.91,
    1.00,
]


def safe_divide(numerator: float, denominator: float) -> float:
    """Avoid division errors in metric calculations."""
    return numerator / denominator if denominator else 0.0


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    """Standardize target values to Y/N labels."""
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {"Y": "Y", "N": "N", "1": "Y", "0": "N", "TRUE": "Y", "FALSE": "N", "YES": "Y", "NO": "N"}
    normalized = normalized.map(mapping)
    if normalized.isna().any():
        invalid = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported values found in {TARGET}: {invalid}")
    return normalized


def get_positive_class_index(model: CatBoostClassifier) -> int:
    """Find the probability column for the positive class."""
    classes = [str(value).strip().upper() for value in model.classes_]
    if "Y" in classes:
        return classes.index("Y")
    if "1" in classes:
        return classes.index("1")
    return 1


def prepare_features(df: pd.DataFrame, model: CatBoostClassifier) -> tuple[pd.DataFrame, list[str]]:
    """Align input columns to the trained model schema."""
    model_features = list(model.feature_names_)
    missing = [feature for feature in model_features if feature not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing model features: {missing}")

    feature_df = df[model_features].copy()
    cat_indices = list(model.get_cat_feature_indices())
    cat_features = [model_features[index] for index in cat_indices]

    for column in model_features:
        if column in cat_features:
            feature_df[column] = feature_df[column].fillna("unknown").astype(str).str.strip().str.lower()
            feature_df[column] = feature_df[column].replace({"": "unknown", "nan": "unknown", "none": "unknown"})
        else:
            feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")

    return feature_df, cat_features


def build_threshold_analysis(y_true_numeric: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    """Calculate confusion metrics across thresholds."""
    rows = []
    total = len(y_true_numeric)
    for threshold in CHART_THRESHOLDS:
        y_pred = (y_score >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true_numeric, y_pred, labels=[0, 1]).ravel()
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        specificity = safe_divide(tn, tn + fp)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        rows.append(
            {
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "tn_pct": safe_divide(tn, total) * 100,
                "fp_pct": safe_divide(fp, total) * 100,
                "fn_pct": safe_divide(fn, total) * 100,
                "tp_pct": safe_divide(tp, total) * 100,
                "accuracy": safe_divide(tp + tn, total),
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "balanced_accuracy": (recall + specificity) / 2,
                "fp_tp_ratio": safe_divide(fp, tp) if tp else math.nan,
            }
        )
    return pd.DataFrame(rows)


def create_threshold_chart(threshold_df: pd.DataFrame, chart_path: Path) -> None:
    """Save the stacked threshold trade-off chart."""
    fig, ax1 = plt.subplots(figsize=(12, 8))
    ax2 = ax1.twinx()
    x = np.arange(len(threshold_df))
    width = 0.82

    fn_pct = threshold_df["fn_pct"].to_numpy()
    fp_pct = threshold_df["fp_pct"].to_numpy()
    tn_pct = threshold_df["tn_pct"].to_numpy()
    tp_pct = threshold_df["tp_pct"].to_numpy()

    b1 = ax1.bar(x, fn_pct, width=width, color="orange", edgecolor="white", label="fn")
    b2 = ax1.bar(x, fp_pct, width=width, bottom=fn_pct, color="red", edgecolor="white", label="fp")
    b3 = ax1.bar(x, tn_pct, width=width, bottom=fn_pct + fp_pct, color="#2E86DE", edgecolor="white", label="tn")
    b4 = ax1.bar(x, tp_pct, width=width, bottom=fn_pct + fp_pct + tn_pct, color="green", edgecolor="white", label="tp")
    (line,) = ax2.plot(x, threshold_df["fp_tp_ratio"], color="navy", linewidth=2, label="fp/tp")

    ax2.set_yscale("log")
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Percentage of Total Test Samples (%)", fontsize=12)
    ax1.set_xlabel("Threshold", fontsize=12)
    ax1.set_title("Threshold Analysis: Trade-off between Catch Rate and False Alarms", fontsize=16)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{value:g}" for value in threshold_df["threshold"]], rotation=0, fontsize=11)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax2.set_ylabel("FP/TP Ratio (Log Scale)", fontsize=12)

    counts_legend = ax1.legend(
        handles=[b1, b2, b3, b4],
        title="Counts",
        loc="upper left",
        bbox_to_anchor=(1.05, 0.84),
        fontsize=11,
        title_fontsize=12,
    )
    ax1.add_artist(counts_legend)
    ax2.legend(handles=[line], title="Ratio", loc="upper left", bbox_to_anchor=(1.05, 0.62), fontsize=11, title_fontsize=12)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def create_confusion_matrix_plot(cm: np.ndarray, output_path: Path) -> None:
    """Save the confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        cbar=True,
        xticklabels=["Predicted N", "Predicted Y"],
        yticklabels=["Actual N", "Actual Y"],
        annot_kws={"size": 16, "weight": "bold"},
        linewidths=1.5,
        linecolor="white",
    )
    plt.title("Confusion Matrix at Selected Threshold", fontsize=14, pad=14)
    plt.xlabel("Model Prediction")
    plt.ylabel("Ground Truth")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    """Load model, score data, and write evaluation outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    df = pd.read_csv(DATA_PATH)
    df[TARGET] = normalize_binary_labels(df[TARGET])

    X, cat_features = prepare_features(df, model)
    y_true_numeric = (df[TARGET] == "Y").astype(int).to_numpy()
    scores = model.predict_proba(Pool(X, cat_features=cat_features))[:, get_positive_class_index(model)]

    threshold_df = build_threshold_analysis(y_true_numeric, scores)
    threshold_df.to_csv(OUTPUT_DIR / "threshold_analysis.csv", index=False)
    create_threshold_chart(threshold_df, OUTPUT_DIR / "threshold_graph.png")

    y_pred = (scores >= DEFAULT_THRESHOLD).astype(int)
    cm = confusion_matrix(y_true_numeric, y_pred, labels=[0, 1])
    pd.DataFrame(cm, index=["actual_N", "actual_Y"], columns=["predicted_N", "predicted_Y"]).to_csv(
        OUTPUT_DIR / "confusion_matrix.csv"
    )
    create_confusion_matrix_plot(cm, OUTPUT_DIR / "confusion_matrix.png")

    metrics = {
        "rows_evaluated": len(df),
        "threshold": DEFAULT_THRESHOLD,
        "roc_auc": roc_auc_score(y_true_numeric, scores) if len(np.unique(y_true_numeric)) == 2 else math.nan,
        "average_precision": average_precision_score(y_true_numeric, scores)
        if len(np.unique(y_true_numeric)) == 2
        else math.nan,
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    print(f"Saved outputs in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
