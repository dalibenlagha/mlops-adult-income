"""Atelier 3: Prefect pipeline for the Adult income project.

Usage:
    python pipeline_prefect.py --flow all        # code checks + prepare + train + save + evaluate
    python pipeline_prefect.py --flow train      # prepare + train (alias: entrainement)
    python pipeline_prefect.py --flow evaluate   # load + evaluate
    python pipeline_prefect.py --flow code       # format + quality + security + tests
"""

import argparse
import subprocess  # nosec B404
import sys

from prefect import flow, task
from prefect.cache_policies import NO_CACHE

from main import evaluate_model, load_model, prepare_data, save_model, train_model

# Only our project files are formatted / checked (not venv/)
PROJECT_FILES = ["model_pipeline.py", "main.py", "pipeline_prefect.py"]


def run(cmd):
    """Run a command with the venv's Python and stop the pipeline if it fails."""
    print("$", " ".join(cmd))
    subprocess.run([sys.executable, "-m", *cmd], check=True)  # nosec B603


# ---------- Tasks: install + code checks ----------
@task(name="Installer les dependances")
def install_dependencies():
    run(["pip", "install", "-q", "-r", "requirements.txt"])


@task(name="Formatage du code")
def format_code():
    run(["black", "--line-length", "100", *PROJECT_FILES])


@task(name="Qualite du code")
def check_quality():
    run(["flake8", "--max-line-length", "100", "--extend-ignore", "E203", *PROJECT_FILES])


@task(name="Securite du code")
def check_security():
    run(["bandit", "-q", *PROJECT_FILES])


@task(name="Tests unitaires")
def run_unit_tests():
    run(["pytest", "-q", "tests"])


# ---------- Tasks: data + model ----------
@task(name="Preparation des donnees", cache_policy=NO_CACHE)
def prepare_task():
    X_train, X_test, y_train, y_test = prepare_data("adult.data", "adult.test")
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


@task(name="Entrainement du modele", cache_policy=NO_CACHE)
def train_task(X_train, y_train):
    return train_model(X_train, y_train)


@task(name="Sauvegarde du modele", cache_policy=NO_CACHE)
def save_task(model):
    save_model(model)


@task(name="Chargement du modele", cache_policy=NO_CACHE)
def load_task():
    return load_model()


@task(name="Evaluation du modele", cache_policy=NO_CACHE)
def evaluate_task(model, X_test, y_test):
    return evaluate_model(model, X_test, y_test)


@task(name="Prediction", cache_policy=NO_CACHE)
def predict_task(model, X_test, n=5):
    preds = model.predict(X_test.head(n))
    labels = [">50K" if p == 1 else "<=50K" for p in preds]
    print(f"Predictions for the first {n} people: {labels}")
    return labels


# ---------- Flows ----------
@flow(name="code", log_prints=True)
def code_flow():
    format_code()
    check_quality()
    check_security()
    run_unit_tests()


@flow(name="train", log_prints=True)
def train_flow():
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    return model


@flow(name="evaluate", log_prints=True)
def evaluate_flow():
    _, X_test, _, y_test = prepare_task()
    model = load_task()
    metrics = evaluate_task(model, X_test, y_test)
    predict_task(model, X_test)
    return metrics


@flow(name="all", log_prints=True)
def all_flow():
    install_dependencies()
    code_flow()
    X_train, X_test, y_train, y_test = prepare_task()
    model = train_task(X_train, y_train)
    save_task(model)
    metrics = evaluate_task(model, X_test, y_test)
    predict_task(model, X_test)
    return metrics


FLOWS = {
    "all": all_flow,
    "train": train_flow,
    "entrainement": train_flow,
    "evaluate": evaluate_flow,
    "code": code_flow,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prefect pipeline")
    parser.add_argument("--flow", choices=FLOWS.keys(), default="all")
    FLOWS[parser.parse_args().flow]()
