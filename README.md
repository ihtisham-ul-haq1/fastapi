# Body Metrics & Gender Prediction API

Starter Project to upload ML apis on Render

Render Link: https://ml-al-on-multiplatform.onrender.com/

A FastAPI-based machine learning API that provides predictions for body measurements and gender detection from images.

## 🚀 Features

- **Body Measurements Prediction**: Predict chest, waist, hips, and shoulder measurements based on height, weight, and sex
- **Gender Prediction**: Detect gender from uploaded images
- **Multiple Deployment Options**: Support for standard deployment and serverless functions

## 📋 Requirements

### Main Application
```
fastapi
pandas
joblib
uvicorn[standard]
numpy
scikit-learn
opencv-python-headless
pillow
onnxruntime
python-multipart
```

### Serverless API (Vercel/AWS Lambda)
```
fastapi
pandas
joblib
uvicorn[standard]
numpy
mangum
```

## 🔧 Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fastapi
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # source venv/bin/activate  # On Unix/macOS
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:
   ```bash
   uvicorn index:app --reload
   ```

## 📚 API Documentation

Once the application is running, you can access the interactive API documentation at:
- http://localhost:8000/docs (Swagger UI)
- http://localhost:8000/redoc (ReDoc)

### Endpoints

#### Home
```
GET /
```
Returns a welcome message.

#### Body Measurements Prediction
```
POST /predict
```
Predicts body measurements based on sex, height, and weight.

**Request Body:**
```json
{
  "sex": "male",
  "height": 180.0,
  "weight": 75.0
}
```

**Response:**
```json
{
  "input": {
    "sex": "male",
    "height_cm": 180.0,
    "weight_kg": 75.0
  },
  "predicted_measurements_cm": {
    "chest": 100.5,
    "waist": 85.2,
    "hips": 95.7,
    "shoulder": 45.3
  }
}
```

#### Gender Prediction
```
POST /predict_gender
```
Predicts gender from an uploaded image.

**Request:**
- Form data with a file upload field named "file"

**Response:**
```json
{
  "gender": "Male"
}
```

## 🚢 Deployment

### Standard Deployment (e.g., on Render)

The main application in the root directory is configured for standard deployment platforms like Render.

### Serverless Deployment (Vercel/AWS Lambda)

The application in the `api` directory is configured for serverless deployment:

- **Vercel**: Uses the Mangum handler to adapt FastAPI for serverless functions
- **AWS Lambda**: Can be deployed as a Lambda function with API Gateway

## 🧠 Machine Learning Models

The application uses several pre-trained models:

- `model.pkl`: Regression model for body measurements prediction
- `label_encoder.pkl`: Label encoder for sex input
- `genderage.onnx`: ONNX model for gender and age prediction from images

## 🛠️ Project Structure

```
fastapi/
├── index.py                # Main FastAPI application
├── requirements.txt        # Dependencies for standard deployment
├── model.pkl               # Body measurements prediction model
├── label_encoder.pkl       # Label encoder for sex input
├── genderage.onnx          # Gender prediction model
└── api/                    # Serverless deployment configuration
    ├── index.py            # FastAPI application for serverless
    ├── requirements.txt    # Dependencies for serverless deployment
    ├── model.pkl           # Body measurements prediction model
    └── label_encoder.pkl   # Label encoder for sex input
```

## 📝 License

[MIT License](LICENSE)

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📊 Example Usage

### Using cURL

#### Body Measurements Prediction
```bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'Content-Type: application/json' \
  -d '{
  "sex": "female",
  "height": 165.0,
  "weight": 60.0
}'
```

#### Gender Prediction
```bash
curl -X 'POST' \
  'http://localhost:8000/predict_gender' \
  -F 'file=@/path/to/your/image.jpg'
```

### Using Python Requests
```python
# First install requests if not already installed:
# pip install requests

import requests

# Body measurements prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"sex": "male", "height": 180.0, "weight": 75.0}
)
print(response.json())

# Gender prediction
with open("path/to/image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict_gender",
        files={"file": f}
    )
print(response.json())


```