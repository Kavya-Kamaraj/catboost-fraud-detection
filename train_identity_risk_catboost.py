from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "dataset" / "identity_risk_train.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "catboost_identity_risk_model.cbm"
TARGET = "target_label"
RANDOM_SEED = 42

CATEGORICAL_FEATURES = [
    "identity_name_risk_level",
    "phone_risk_level",
    "address_risk_level",
    "phone_email_match",
    "phone_dob_match",
    "phone_city_match",
    "phone_zipcode_match",
    "phone_state_match",
    "phone_name_match",
    "phone_address_match",
    "overall_identity_risk_level",
]


def normalize_binary_labels(series: pd.Series) -> pd.Series:
    """Standardize target values to Y/N labels."""
    normalized = series.astype(str).str.strip().str.upper()
    mapping = {
        "Y": "Y",
        "N": "N",
        "1": "Y",
        "0": "N",
        "TRUE": "Y",
        "FALSE": "N",
        "YES": "Y",
        "NO": "N",
    }
    normalized = normalized.map(mapping)
    if normalized.isna().any():
        invalid = sorted(series[normalized.isna()].astype(str).unique().tolist())
        raise ValueError(f"Unsupported values found in {TARGET}: {invalid}")
    return normalized


def load_training_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load, clean, and validate the training CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Training dataset not found: {path}")

    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' not found in {path}")

    for column in df.columns:
        if column == TARGET:
            continue
        if df[column].dtype == "object" or df[column].dtype == "bool":
            df[column] = df[column].astype(str).str.strip().str.lower()

    df[TARGET] = normalize_binary_labels(df[TARGET])
    return df.dropna().reset_index(drop=True)


def build_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select model inputs and exclude metadata columns."""
    ignored_columns = {
        "record_id",
        "event_timestamp",
        "event_timestamp_utc",
        "event_month",
        "source_group",
        "applicant_id",
        "event_id",
    }
    return [column for column in df.columns if column not in {TARGET, *ignored_columns}]


def train_model() -> CatBoostClassifier:
    """Train CatBoost and save the fitted model artifact."""
    df = load_training_data()
    feature_cols = build_feature_columns(df)
    categorical_features = [column for column in CATEGORICAL_FEATURES if column in feature_cols]

    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    train_pool = Pool(X_train, y_train, cat_features=categorical_features)
    test_pool = Pool(X_test, y_test, cat_features=categorical_features)

    model = CatBoostClassifier(
        task_type="GPU",
        auto_class_weights="Balanced",
        use_best_model=True,
        iterations=10000,
        depth=6,
        learning_rate=0.01,
        random_seed=RANDOM_SEED,
        verbose=100,
        loss_function="Logloss",
        eval_metric="AUC",
    )

    print(f"Rows loaded: {len(df)}")
    print(f"Training rows: {len(X_train)}")
    print(f"Test rows: {len(X_test)}")
    print(f"Features: {len(feature_cols)}")
    print("Starting CatBoost training...")

    model.fit(train_pool, eval_set=test_pool, early_stopping_rounds=500)

    y_train_score = model.predict_proba(X_train)[:, 1]
    y_test_score = model.predict_proba(X_test)[:, 1]
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(f"Train AUC: {roc_auc_score(y_train, y_train_score):.4f}")
    print(f"Test AUC: {roc_auc_score(y_test, y_test_score):.4f}")
    print(f"Train Accuracy: {accuracy_score(y_train, y_train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    print(f"Model saved: {MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_model()
