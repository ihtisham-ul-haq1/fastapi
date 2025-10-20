import sys
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

sys.dont_write_bytecode = True

app = FastAPI()

# Load pretrained model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

# Initialize FER detector
# detector = FER(mtcnn=True)

# ✅ Define input model
class BodyInput(BaseModel):
    sex: str
    height: float
    weight: float


@app.get("/")
async def home():
    return JSONResponse({"message": "Hello from FastAPI on Render!"})


@app.post("/predict")
async def predict_measurements(data: BodyInput):
    """
    Predict body measurements based on height, weight, and sex.
    """
    sex_input = data.sex.lower()
    if sex_input not in ["male", "female"]:
        return {"error": "Sex must be 'male' or 'female'"}

    # Encode and predict
    sex_encoded = le.transform([sex_input])[0]
    features = np.array([[sex_encoded, data.height, data.weight]])
    prediction = model.predict(features)[0]

    # Return results
    return {
        "input": {
            "sex": sex_input,
            "height_cm": data.height,
            "weight_kg": data.weight
        },
        "predicted_measurements_cm": {
            "chest": round(prediction[0], 1),
            "waist": round(prediction[1], 1),
            "hips": round(prediction[2], 1),
            "shoulder": round(prediction[3], 1)
        }
    }

#
# @app.post("/predict_emotion")
# async def predict_emotion(file: UploadFile = File(...)):
#     # Read image as array
#     image = np.frombuffer(await file.read(), np.uint8)
#     image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#
#     result = detector.detect_emotions(image)
#     if not result:
#         return {"error": "No face detected"}
#
#     emotions = result[0]["emotions"]
#     dominant = max(emotions, key=emotions.get)
#
#     return {
#         "dominant_emotion": dominant,
#         "emotions": emotions
#     }


# Required for Vercel
handler = Mangum(app)
