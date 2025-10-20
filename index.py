import sys
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import onnxruntime as ort
import mediapipe as mp
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
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)

# Load ONNX gender model
# You need a small gender classifier in ONNX format
# Input: 64x64 RGB image normalized [0,1]
# Output: [male_prob, female_prob]
onnx_session = ort.InferenceSession("gender_model.onnx")
GENDER_LIST = ["male", "female"]

def preprocess_face(face_img: np.ndarray):
    """Resize and normalize the face image for ONNX model"""
    face_img = Image.fromarray(face_img)
    face_img = face_img.resize((64, 64))
    face_array = np.array(face_img).astype(np.float32) / 255.0
    face_array = np.transpose(face_array, (2, 0, 1))  # HWC → CHW if model needs
    face_array = np.expand_dims(face_array, axis=0)
    return face_array

@app.post("/predict_gender")
async def predict_gender_api(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)

        # Detect faces
        results = face_detection.process(image_np)
        if not results.detections:
            return JSONResponse({"error": "No face detected"}, status_code=400)

        # Use first detected face
        detection = results.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        h, w, _ = image_np.shape
        x1 = int(bboxC.xmin * w)
        y1 = int(bboxC.ymin * h)
        x2 = x1 + int(bboxC.width * w)
        y2 = y1 + int(bboxC.height * h)

        face_img = image_np[y1:y2, x1:x2]

        # Preprocess and predict
        input_tensor = preprocess_face(face_img)
        pred = onnx_session.run(None, {onnx_session.get_inputs()[0].name: input_tensor})[0]
        gender = GENDER_LIST[int(np.argmax(pred))]

        return JSONResponse({"gender": gender})

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