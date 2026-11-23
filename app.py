"""Expose the Adult income model as a REST API with FastAPI.

Run:   uvicorn app:app --reload
Docs:  http://127.0.0.1:8000/docs
"""

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from model_pipeline import COLUMNS, load_model, transform

app = FastAPI(title="Adult Income Prediction API", version="1.0")
model = load_model("model.joblib")


class Person(BaseModel):
    """One person's census information (same columns as adult.data)."""

    age: int = Field(..., ge=17, le=90, examples=[39])
    workclass: str = Field(..., examples=["State-gov"])
    education_num: int = Field(..., ge=1, le=16, examples=[13])
    marital_status: str = Field(..., examples=["Never-married"])
    occupation: str = Field(..., examples=["Adm-clerical"])
    relationship: str = Field(..., examples=["Not-in-family"])
    race: str = Field(..., examples=["White"])
    sex: str = Field(..., examples=["Male"])
    capital_gain: int = Field(0, ge=0, examples=[2174])
    capital_loss: int = Field(0, ge=0, examples=[0])
    hours_per_week: int = Field(..., ge=1, le=99, examples=[40])
    native_country: str = Field(..., examples=["United-States"])


def to_features(person: Person) -> pd.DataFrame:
    """Apply exactly the same preparation as in training, then align the columns."""
    row = {
        "age": person.age,
        "workclass": person.workclass,
        "fnlwgt": 0,  # dropped by transform()
        "education": "",  # dropped by transform()
        "education-num": person.education_num,
        "marital-status": person.marital_status,
        "occupation": person.occupation,
        "relationship": person.relationship,
        "race": person.race,
        "sex": person.sex,
        "capital-gain": person.capital_gain,
        "capital-loss": person.capital_loss,
        "hours-per-week": person.hours_per_week,
        "native-country": person.native_country,
        "income": "<=50K",  # placeholder, not used for prediction
    }
    df = transform(pd.DataFrame([row], columns=COLUMNS), mode_country="United-States")
    if df.empty:
        raise HTTPException(status_code=422, detail="'Never-worked' is not supported by the model")
    return df.drop(columns=["income"]).reindex(columns=model.feature_names_in_, fill_value=0)


@app.get("/")
def home():
    return {"message": "Adult Income API is running. Go to /docs to test it."}


@app.post("/predict")
def predict(person: Person):
    X = to_features(person)
    proba = float(model.predict_proba(X)[0, 1])
    return {
        "prediction": ">50K" if proba >= 0.5 else "<=50K",
        "probability_over_50K": round(proba, 3),
    }
