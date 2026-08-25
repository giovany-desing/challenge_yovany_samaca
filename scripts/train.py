"""Entrena el modelo una sola vez y persiste el artefacto en disco."""
import pandas as pd
from pathlib import Path
from challenge.model import DelayModel

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
ARTIFACT_PATH = BASE_DIR / "challenge" / "artifacts" / "model.joblib"

def main():
    ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH, low_memory=False)
    model = DelayModel()
    features, target = model.preprocess(df, target_column="delay")
    model.fit(features, target)
    model.save(str(ARTIFACT_PATH))
    print(f"Modelo entrenado y guardado en {ARTIFACT_PATH}")

if __name__ == "__main__":
    main()