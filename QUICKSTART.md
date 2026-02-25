# Quick Start Guide - Women Risk Predictor

Follow these steps to get your ML project up and running!

## 🚀 Quick Setup (5 minutes)

### Step 1: Install Dependencies (1 minute)
```bash
pip install -r requirements.txt
```

### Step 2: Run the Complete Pipeline (3-5 minutes)
```bash
cd backend
python run_pipeline.py
```

This single command will:
- ✅ Load and clean your dataset
- ✅ Perform feature engineering
- ✅ Train multiple ML models
- ✅ Compare and select the best model
- ✅ Perform hyperparameter tuning
- ✅ Save the trained model
- ✅ Generate visualizations

### Step 3: Start the API Server
```bash
python app.py
```

### Step 4: Access the Dashboard
Open your browser and visit:
```
http://127.0.0.1:5000/dashboard
```

Or visit the API documentation at:
```
http://127.0.0.1:5000
```

### Step 5: Test the API (Optional)
Open a new terminal and run:
```bash
python test_api.py
```

## 🧪 Quick Test with cURL

```bash
# Test health
curl http://127.0.0.1:5000/health

# Test prediction (adjust values based on your dataset)
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "occupation": 0,
    "location": 1,
    "time_of_day": 3,
    "public_transport_usage": 1,
    "past_incidents": 2
  }'
```

## 📁 What Gets Generated

After running the pipeline, you'll have:

```
backend/
├── data/
│   ├── women_risk.csv                 # Original dataset
│   ├── women_risk_cleaned.csv         # Cleaned data
│   ├── women_risk_processed.csv       # Processed data
│   ├── correlation_heatmap.png        # Feature correlations
│   ├── target_correlation.png         # Target correlations
│   ├── model_comparison.png           # Model comparison chart
│   ├── confusion_matrix_*.png         # Confusion matrices
│   └── roc_curve_*.png               # ROC curves
├── models/
│   ├── women_risk_model.pkl          # Trained model
│   ├── scaler.pkl                    # Feature scaler
│   ├── label_encoders.pkl            # Categorical encoders
│   └── model_info.txt                # Model details
```

## 🎯 Running Individual Steps

If you want to run steps separately:

```bash
cd backend/scripts

# Step 1: Data preparation
python data_preparation.py

# Step 2: Feature engineering
python feature_engineering.py

# Step 3: Model training
python model_training.py
```

## ⚡ Troubleshooting

### Issue: "Dataset not found"
**Solution:** Make sure your dataset is at:
```
backend/data/women_risk.csv
```

### Issue: "ModuleNotFoundError"
**Solution:** Install dependencies:
```bash
pip install -r requirements.txt
```

### Issue: "Port already in use"
**Solution:** Change the port in app.py or stop the other process:
```python
app.run(debug=True, port=5001)  # Use different port
```

## 📊 View Results

1. **Check model performance:**
   - Open `backend/models/model_info.txt`

2. **View visualizations:**
   - Check PNG files in `backend/data/` folder

3. **Test the API:**
   - Visit `http://127.0.0.1:5000` in your browser
   - Or use the test script: `python test_api.py`

## 🎉 You're Done!

Your ML project is now complete with:
- ✅ Data preprocessing
- ✅ Feature engineering
- ✅ Model training & evaluation
- ✅ Working REST API
- ✅ Interactive web interface

Now you can make predictions about harassment risk through the API!

---

**Need help?** Check the main README.md for detailed documentation.
