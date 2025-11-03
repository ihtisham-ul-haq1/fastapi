import sys
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import onnxruntime as ort
import cv2
import os
import io
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

sys.dont_write_bytecode = True
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load pretrained model and label encoder
model = joblib.load("model.pkl")
le = joblib.load("label_encoder.pkl")

class BodyInput(BaseModel):
    sex: str
    height: float
    weight: float

@app.get("/")
async def home():
    return JSONResponse({"message": "Hello from FastAPI on Render!"})

@app.post("/predict")
async def predict_measurements(data: BodyInput):

    sex_input = data.sex.lower()
    if sex_input not in ["male", "female"]:
        return {"error": "Sex must be 'male' or 'female'"}

    sex_encoded = le.transform([sex_input])[0]
    features = np.array([[sex_encoded, data.height, data.weight]])
    prediction = model.predict(features)[0]
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
    print(outputs)
    out = outputs[0][0]  # shape (3,)

    # Gender
    gender = "Male" if out[0] > out[1] else "Female"

    # Age
    age = int(out[2])
    return gender, int(age)

# -------------------------------
# FastAPI endpoint
# -------------------------------
@app.post("/predict_gender")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image_tensor = preprocess_image(image_bytes)
        outputs = session.run(None, {input_name: image_tensor})
        out = outputs[0][0]
        print(outputs)
        gender = "Male" if out[0] > out[1] else "Female"
        age = float(out[2]) * 100

        return JSONResponse({"gender": gender})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)