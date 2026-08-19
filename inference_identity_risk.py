import argparse
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "catboost_identity_risk_model.cbm"
DEFAULT_INPUT_PATH = BASE_DIR / "dataset" / "identity_risk_train.csv"
DEFAULT_OUTPUT_PATH = BASE_DIR / "outputs" / "inference_predictions.csv"
DEFAULT_THRESHOLD = 0.55


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
        raise ValueError(f"Input file is missing model features: {missing}")

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


def run_inference(model_path: Path, input_path: Path, output_path: Path, threshold: float) -> None:
    """Score a CSV and save predictions."""
    model = CatBoostClassifier()
    model.load_model(str(model_path))

    df = pd.read_csv(input_path)
    X, cat_features = prepare_features(df, model)
    scores = model.predict_proba(Pool(X, cat_features=cat_features))[:, get_positive_class_index(model)]

    output_df = df.copy()
    output_df["model_score"] = scores
    output_df["model_prediction"] = ["Y" if score >= threshold else "N" for score in scores]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False)
    print(f"Saved predictions: {output_path}")


def parse_args() -> argparse.Namespace:
    """Read command-line options."""
    parser = argparse.ArgumentParser(description="Run CatBoost inference on a CSV file.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_inference(args.model_path, args.input_path, args.output_path, args.threshold)
