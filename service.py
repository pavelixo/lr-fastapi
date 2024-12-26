from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load
import numpy as np

model = load("model.pkl")

app = FastAPI()

class PredictionRequest(BaseModel):
    feature: float

class PredictionResponse(BaseModel):
    prediction: float

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    prediction = model.predict(np.array([[request.feature]]))
    return PredictionResponse(prediction=prediction[0])

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
