from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from identify import random_forest, classify_one, removal_files

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

df = pd.read_csv("./GTZAN/features_30_sec.csv")
df_removals = df[~df["filename"].isin(removal_files)].copy()
df_removals = df_removals.drop(columns=["filename"])

removed_forest, scaler, low_base = random_forest(df_removals, "Modified Dataset")

class ClassifyRequest(BaseModel):
    filename: str

@app.post("/classify")
def classify(body: ClassifyRequest):
    prediction, actual_label = classify_one(body.filename, removed_forest, scaler)
    return {
        "prediction": prediction[0],
        "actual": actual_label,
        "low_baseline": low_base,
    }