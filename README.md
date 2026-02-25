# Women Risk Predictor - Machine Learning Project

A machine learning-based system for predicting harassment risk levels for women based on various factors.

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict harassment risk, including:
- Data preprocessing and cleaning
- Feature engineering and selection
- Multiple model training and comparison
- Hyperparameter tuning
- Flask REST API for predictions

## 🗂️ Project Structure

```
women_risk_predictor/
├── backend/
│   ├── data/                          # Data files
│   │   ├── women_risk.csv            # Original dataset
│   │   ├── women_risk_cleaned.csv    # Cleaned dataset
│   │   └── women_risk_processed.csv  # Processed dataset
│   ├── models/                        # Trained models
│   │   ├── women_risk_model.pkl      # Final trained model
│   │   ├── scaler.pkl                # Feature scaler
│   │   ├── label_encoders.pkl        # Label encoders
│   │   └── model_info.txt            # Model information
│   ├── scripts/                       # Python scripts
│   │   ├── data_preparation.py       # Step 2: Data preparation
│   │   ├── feature_engineering.py    # Step 3: Feature engineering
│   │   └── model_training.py         # Steps 4-7: Training & evaluation
│   └── app.py                         # Flask API application
└── frontend/
    └── templates/
        └── index.html                 # API home page

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## 📊 Running the ML Pipeline

### Step 1: Prepare the Data

```bash
cd backend/scripts
python data_preparation.py
```

This script will:
- Load the dataset
- Explore and analyze the data
- Check for missing values
- Remove duplicates
- Encode categorical variables
- Save the cleaned dataset

### Step 2: Feature Engineering

```bash
python feature_engineering.py
```

This script will:
- Perform correlation analysis
- Create new features
- Scale numeric features
- Generate visualizations
- Save the processed dataset

### Step 3: Train the Model

```bash
python model_training.py
```

This script will:
- Train multiple classification models
- Compare model performances
- Perform hyperparameter tuning
- Evaluate the best model
- Save the trained model

## 🌐 Running the Flask API

### Start the API Server

```bash
cd backend
python app.py
```

The API will be available at: `http://127.0.0.1:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

#### 2. Model Information
```bash
GET /model_info
```

#### 3. Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "age": 25,
  "occupation": "Student",
  "location": "Urban",
  "time_of_day": "Night",
  "public_transport_usage": 1,
  "past_incidents": 2
}
```

**Response:**
```json
{
  "risk": 1,
  "risk_label": "High Risk",
  "probability": 0.8542,
  "message": "High risk detected. Please be cautious...",
  "status": "success"
}
```

#### 4. Batch Prediction
```bash
POST /predict_batch
Content-Type: application/json

{
  "data": [
    {"age": 25, "occupation": "Student", ...},
    {"age": 30, "occupation": "Working", ...}
  ]
}
```

## 🧪 Testing the API

### Using cURL:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "occupation": "Student",
    "location": "Urban",
    "time_of_day": "Night",
    "public_transport_usage": 1,
    "past_incidents": 2
  }'
```

### Using Python:

```python
import requests
import json

url = "http://127.0.0.1:5000/predict"
data = {
    "age": 25,
    "occupation": "Student",
    "location": "Urban",
    "time_of_day": "Night",
    "public_transport_usage": 1,
    "past_incidents": 2
}

response = requests.post(url, json=data)
print(json.dumps(response.json(), indent=2))
```

## 📈 Models Trained

The project trains and compares the following models:
1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier
4. Gradient Boosting Classifier
5. Support Vector Machine (SVM)

The best-performing model is selected and fine-tuned using GridSearchCV.

## 📊 Evaluation Metrics

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

## 📁 Dataset

The dataset (`harassment_data_no_missing.csv`) should contain columns such as:
- `age`: Age of the person
- `occupation`: Occupation category
- `location`: Location type (Urban/Rural)
- `time_of_day`: Time of day (Morning/Afternoon/Evening/Night)
- `public_transport_usage`: Binary (0/1)
- `past_incidents`: Number of past incidents
- `risk`: Target variable (0 = Low Risk, 1 = High Risk)

## 🔧 Troubleshooting

### Issue: Model not found
**Solution:** Run the model training script first:
```bash
python scripts/model_training.py
```

### Issue: Missing data files
**Solution:** Ensure the dataset is placed in `backend/data/women_risk.csv`

### Issue: Package import errors
**Solution:** Install all dependencies:
```bash
pip install -r requirements.txt
```

## ⚠️ Important Notes

- This is a prototype ML system for educational purposes
- Always prioritize personal safety regardless of predictions
- The model's predictions should be used as guidance, not absolute truth
- Regular model retraining is recommended as new data becomes available

## 🤝 Contributing

This is a learning project. Feel free to improve it!

## 📝 License

This project is for educational purposes.

## 👤 Author

ML Project - Women Risk Predictor
Date: February 2026

---

**Stay Safe! 🛡️**
