from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import torch
from torch import nn
import joblib

scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")


class AbaloneModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


input_dim = len(feature_columns)
model = AbaloneModel(input_dim)
model.load_state_dict(torch.load("abalone_model.pth", map_location=torch.device("cpu")))
model.eval()

app = FastAPI(
    title="Abalone Age Prediction Service",
    description="Predicts abalone rings and estimated age from physical measurements.",
    version="1.0"
)


class AbaloneInput(BaseModel):
    sex: str
    length: float
    diameter: float
    height: float
    whole_weight: float
    shucked_weight: float
    viscera_weight: float
    shell_weight: float


def preprocess_input(data: AbaloneInput):
    sex_value = data.sex.upper().strip()

    if sex_value not in ["M", "F", "I"]:
        raise ValueError("sex must be one of: M, F, I")

    raw_df = pd.DataFrame([{
        "Length": data.length,
        "Diameter": data.diameter,
        "Height": data.height,
        "Whole_weight": data.whole_weight,
        "Shucked_weight": data.shucked_weight,
        "Viscera_weight": data.viscera_weight,
        "Shell_weight": data.shell_weight,
        "Sex": sex_value
    }])

    processed_df = pd.get_dummies(raw_df, columns=["Sex"], drop_first=True)
    processed_df = processed_df.reindex(columns=feature_columns, fill_value=0)
    scaled = scaler.transform(processed_df)

    tensor_input = torch.tensor(scaled, dtype=torch.float32)
    return tensor_input


@app.get("/")
def home():
    return {
        "message": "Abalone Age Prediction Service is running.",
        "docs": "/docs",
        "usage": "Send a POST request to /predict with abalone measurements."
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: AbaloneInput):
    try:
        x = preprocess_input(data)

        with torch.no_grad():
            prediction = model(x).item()

        predicted_rings = float(prediction)
        estimated_age = float(predicted_rings + 1.5)

        return {
            "input": {
                "sex": data.sex,
                "length": data.length,
                "diameter": data.diameter,
                "height": data.height,
                "whole_weight": data.whole_weight,
                "shucked_weight": data.shucked_weight,
                "viscera_weight": data.viscera_weight,
                "shell_weight": data.shell_weight
            },
            "predicted_rings": round(predicted_rings, 2),
            "estimated_age_years": round(estimated_age, 2)
        }

    except Exception as e:
        return {"error": str(e)}