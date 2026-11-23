"""Modular ML pipeline for the Adult income project (>50K or <=50K)."""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
CAT_COLS = ["workclass", "occupation", "relationship", "race", "sex", "native-country"]
MARRIED = ["Married-civ-spouse", "Married-AF-spouse"]


def load_data(train_path="adult.data", test_path="adult.test"):
    """Read the raw train and test files and add the column names."""
    train = pd.read_csv(train_path, header=None, names=COLUMNS, skipinitialspace=True)
    test = pd.read_csv(test_path, header=None, names=COLUMNS, skipinitialspace=True, skiprows=1)
    return train, test


def clean_train(df):
    """Cleaning only done on the training data: duplicates and impossible rows."""
    df = df.drop_duplicates()
    errors = ((df["sex"] == "Female") & (df["relationship"] == "Husband")) | (
        (df["sex"] == "Male") & (df["relationship"] == "Wife")
    )
    return df[~errors]


def transform(df, mode_country):
    """Same transformations for train and test (missing values, features, log, target)."""
    df = df.copy()
    df["income"] = df["income"].str.replace(".", "", regex=False)
    df = df[df["workclass"] != "Never-worked"]

    df["workclass"] = df["workclass"].replace(["?", "Without-pay"], "Unknown")
    df["occupation"] = (
        df["occupation"].replace("?", "Unknown").replace("Armed-Forces", "Protective-serv")
    )
    df["native-country"] = df["native-country"].replace("?", mode_country)
    df["native-country"] = np.where(
        df["native-country"] == "United-States", "United-States", "Other"
    )

    df = df.drop(columns=["fnlwgt", "education"])
    df["has_capital_gain"] = (df["capital-gain"] > 0).astype(int)
    df["has_capital_loss"] = (df["capital-loss"] > 0).astype(int)
    df["lives_with_spouse"] = df["marital-status"].isin(MARRIED).astype(int)
    df = df.drop(columns=["marital-status"])

    df["capital-gain"] = np.log1p(df["capital-gain"])
    df["capital-loss"] = np.log1p(df["capital-loss"])
    df["income"] = (df["income"] == ">50K").astype(int)
    return pd.get_dummies(df, columns=CAT_COLS, drop_first=True, dtype=int)


def prepare_data(train_path="adult.data", test_path="adult.test"):
    """Load, clean and encode the data. Returns X_train, X_test, y_train, y_test."""
    train, test = load_data(train_path, test_path)
    train = clean_train(train)
    mode_country = train.loc[train["native-country"] != "?", "native-country"].mode()[0]

    train_final = transform(train, mode_country)
    test_final = transform(test, mode_country).reindex(columns=train_final.columns, fill_value=0)

    X_train, y_train = train_final.drop(columns=["income"]), train_final["income"]
    X_test, y_test = test_final.drop(columns=["income"]), test_final["income"]
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, balanced=False):
    """Train the final model of the project: Random Forest (200 trees, depth 15)."""
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced" if balanced else None,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """Print and return accuracy, precision, recall, F1 and ROC-AUC."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    for name, value in metrics.items():
        print(f"{name:<10}: {value:.3f}")
    print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=["<=50K", ">50K"]))
    return metrics


def save_model(model, path="model.joblib"):
    """Save the trained model with joblib."""
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path="model.joblib"):
    """Load a saved model."""
    return joblib.load(path)
