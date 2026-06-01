# challenge/api.py

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from pathlib import Path

from .model import DelayModel

# ------------------------------
# Robust path resolution
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"

# Known valid categories (will be populated during training)
KNOWN_OPERAS = set()
KNOWN_TIPOVUELO = set()
KNOWN_MES = set(range(1, 13))

model = None

def _load_and_train():
    global model, KNOWN_OPERAS, KNOWN_TIPOVUELO
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    KNOWN_OPERAS = set(df['OPERA'].unique())
    KNOWN_TIPOVUELO = set(df['TIPOVUELO'].unique())
    model = DelayModel()
    # CORRECCIÓN: pasar target_column para entrenamiento
    features, target = model.preprocess(df, target_column="delay")
    model.fit(features, target)

# Train when module loads
_load_and_train()

# ------------------------------
# FastAPI app
# ------------------------------
app = FastAPI()

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int

class PredictRequest(BaseModel):
    flights: List[Flight]

class PredictResponse(BaseModel):
    predict: List[int]

@app.get("/health", status_code=200)
async def get_health():
    return {"status": "OK"}

@app.post("/predict", status_code=200, response_model=PredictResponse)
async def post_predict(request: PredictRequest):
    for flight in request.flights:
        if flight.OPERA not in KNOWN_OPERAS:
            raise HTTPException(status_code=400, detail=f"Unknown OPERA: {flight.OPERA}")
        if flight.TIPOVUELO not in KNOWN_TIPOVUELO:
            raise HTTPException(status_code=400, detail=f"Unknown TIPOVUELO: {flight.TIPOVUELO}")
        if flight.MES not in KNOWN_MES:
            raise HTTPException(status_code=400, detail=f"Invalid MES: {flight.MES} (must be 1-12)")

    df = pd.DataFrame([f.dict() for f in request.flights])
    features = model.preprocess(df)
    predictions = model.predict(features)
    return PredictResponse(predict=predictions)