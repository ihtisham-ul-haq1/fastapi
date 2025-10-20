import sys
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import onnxruntime as ort
import cv2

import io
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
MODEL_PATH = os.path.join(os.path.dirname(__file__), "genderage.onnx")
session = ort.InferenceSession(MODEL_PATH)
input_name = session.get_inputs()[0].name

# -------------------------------
# Helper function
# -------------------------------
def preprocess_image(image_bytes: bytes):
    """Load image, resize and normalize for ONNX model"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, (96, 96))  # model expects 96x96
    image = image.transpose(2, 0, 1).astype(np.float32)  # HWC -> CHW
    image = np.expand_dims(image, axis=0)
    return image

def predict_gender_age(image_tensor: np.ndarray):
    """Run ONNX model inference"""
    outputs = session.run(None, {input_name: image_tensor})
    gender_probs, age = outputs[0][0], outputs[1][0]
    gender = "Male" if gender_probs[0] > gender_probs[1] else "Female"
    return gender, int(age)

# -------------------------------
# FastAPI endpoint
# -------------------------------
@app.post("/predict_gender")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)

        # Predict gender and age
        gender, age = predict_gender_age(image_tensor)

        return JSONResponse({
            "gender": gender,
            "age": age
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

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