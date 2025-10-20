import sys
import numpy as np
import joblib
import pandas as pd
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
from PIL import Image
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



app = FastAPI(title="Lightweight Gender Detection API")

# Load pretrained OpenCV models
# Download these files from OpenCV repo:
# https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy.prototxt
# https://github.com/opencv/opencv_3rdparty/blob/dnn_samples_face_detector/res10_300x300_ssd_iter_140000.caffemodel
# https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/gender_net.caffemodel
# https://github.com/opencv/opencv/blob/master/samples/dnn/face_detector/deploy_gender.prototxt

FACE_PROTO = "deploy.prototxt"
FACE_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"
GENDER_PROTO = "deploy_gender.prototxt"
GENDER_MODEL = "gender_net.caffemodel"
GENDER_LIST = ["male", "female"]

face_net = cv2.dnn.readNet(FACE_MODEL, FACE_PROTO)
gender_net = cv2.dnn.readNet(GENDER_MODEL, GENDER_PROTO)

def detect_face(image):
    """Detect faces and return bounding boxes"""
    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()
    faces = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.6:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            faces.append(box.astype(int))
    return faces

def predict_gender(face_img):
    """Predict gender from a face image"""
    blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), [78.4263377603, 87.7689143744, 114.895847746], swapRB=False)
    gender_net.setInput(blob)
    preds = gender_net.forward()
    gender = GENDER_LIST[preds[0].argmax()]
    return gender

@app.post("/predict_gender")
async def predict_gender_api(file: UploadFile = File(...)):
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        faces = detect_face(image)
        if not faces:
            return JSONResponse({"error": "No face detected"}, status_code=400)

        # Only predict the first detected face
        x1, y1, x2, y2 = faces[0]
        face_img = image[y1:y2, x1:x2]
        gender = predict_gender(face_img)

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