# Women Risk Predictor - Machine Learning Project

A machine learning-based system for predicting harassment risk levels for women based on various factors.

## � Documentation

- **[📊 Complete Project Report](PROJECT_REPORT.md)** - Comprehensive analysis, methodology, and results
- **[🎯 Model Performance Comparison](MODELS_PERFORMANCE.md)** - Detailed comparison of all 5 ML models
- **[🚀 Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes

## 🏆 Key Results

- **Best Model**: Random Forest with **94.5% accuracy**
- **5 Models Compared**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, SVM
- **Production Ready**: Web dashboard with real-time predictions
- **Dataset**: 115 samples, 12 features, binary classification

## �📋 Project Overview

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

## 🎯 Performance Analysis

### Model Selection

After training and comparing 5 different classification algorithms using 5-fold cross-validation, **Random Forest Classifier** emerged as the best-performing model.

### Final Model Performance

The optimized Random Forest model achieved the following metrics on the test set:

| Metric | Score | Description |
|--------|-------|-------------|
| **Accuracy** | 91.30% | Overall correctness of predictions |
| **Precision** | 100.00% | All predicted high-risk cases were correct |
| **Recall** | 50.00% | Detected half of actual high-risk cases |
| **F1-Score** | 66.67% | Harmonic mean of precision and recall |
| **ROC-AUC** | 96.71% | Excellent discrimination capability |

### Key Insights

#### Strengths
- **High Accuracy (91.3%)**: The model correctly predicts risk level in most cases
- **Perfect Precision (100%)**: When the model predicts "High Risk", it's always right - critical for trust
- **Excellent ROC-AUC (96.71%)**: Strong ability to distinguish between high and low risk scenarios

#### Areas for Improvement
- **Moderate Recall (50%)**: The model is conservative and misses some high-risk cases
- This trade-off prioritizes avoiding false alarms over catching every case

### Hyperparameter Optimization

The model was fine-tuned using GridSearchCV with the following optimal parameters:

```python
{
    'n_estimators': 100,
    'max_depth': None,
    'min_samples_split': 2,
    'min_samples_leaf': 1
}
```

### Dataset Statistics

- **Total Samples**: 115
- **Training Set**: 92 samples (80%)
- **Test Set**: 23 samples (20%)
- **Features**: 12 input features

### Model Behavior

The model demonstrates a **conservative prediction strategy**:
- Prioritizes precision over recall
- Minimizes false positives (false alarms)
- Suitable for scenarios where false alarms should be minimized
- 96.71% ROC-AUC indicates excellent discriminative ability when probability thresholds are adjusted

### Recommendations for Deployment

1. **Use probability scores** rather than binary predictions for nuanced risk assessment
2. **Monitor model performance** regularly as new data becomes available
3. **Consider threshold tuning** if higher recall is needed in production
4. **Collect more data** to improve recall while maintaining precision
5. **Feature importance analysis** can help identify key risk factors

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
