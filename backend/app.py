"""
Flask API for Women Risk Prediction
This API provides endpoints for predicting harassment risk based on input features.
"""

from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import numpy as np
import os
from werkzeug.utils import secure_filename
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__, 
            template_folder='frontend/templates',
            static_folder='frontend/static')

# Configuration for file uploads
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

# Load the trained model
MODEL_PATH = "models/women_risk_model.pkl"
SCALER_PATH = "models/scaler.pkl"
ENCODER_PATH = "models/label_encoders.pkl"

try:
    model = joblib.load(MODEL_PATH)
    print(f"✓ Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

try:
    scaler = joblib.load(SCALER_PATH)
    print(f"✓ Scaler loaded successfully from {SCALER_PATH}")
except Exception as e:
    print(f"✗ Error loading scaler: {e}")
    scaler = None

try:
    label_encoders = joblib.load(ENCODER_PATH)
    print(f"✓ Label encoders loaded successfully from {ENCODER_PATH}")
except Exception as e:
    print(f"✗ Error loading label encoders: {e}")
    label_encoders = None

@app.route('/')
def home():
    """Home page - API documentation"""
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    """Dashboard page with questionnaire"""
    return render_template('dashboard.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict harassment risk based on input features.
    
    Expected JSON format:
    {
        "age": 25,
        "occupation": "Student",
        "location": "Urban",
        "time_of_day": "Night",
        "public_transport_usage": 1,
        "past_incidents": 2,
        ...
    }
    
    Returns:
    {
        "risk": 1,
        "risk_label": "High Risk",
        "probability": 0.85,
        "message": "High risk detected. Please take necessary precautions."
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first.",
                "status": "error"
            }), 500
        
        # Get JSON data from request
        data = request.get_json()
        
        if not data:
            return jsonify({
                "error": "No data provided. Please send JSON data.",
                "status": "error"
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Add Timestamp column if not present (using a default value)
        if 'Timestamp' not in df.columns:
            df.insert(0, 'Timestamp', 0)  # Add Timestamp as first column
        
        # Ensure correct column order to match training data
        expected_columns = [
            'Timestamp',
            '1. What is your age group?',
            '2. What is your occupation?',
            '3. At what time of day did the incident occur?',
            '4. Where did the incident occur?',
            '5. How crowded was the location at the time of the incident?',
            '6. What was the lighting condition in the area?',
            '7. Was any form of security present at the location?',
            '8. Were you familiar with the area where the incident occurred?',
            '9. What type of harassment did you experience?',
            '10. How often have you experienced harassment in similar situations?',
            '11. How safe did you feel during the incident?'
        ]
        
        # Reorder columns to match expected order
        df = df[expected_columns]
        
        # Apply label encoding if encoders are available
        if label_encoders:
            for col, encoder in label_encoders.items():
                # Skip Timestamp column as it's not used for prediction
                if col == 'Timestamp':
                    continue
                    
                if col in df.columns:
                    try:
                        # Check if value is in the encoder's classes
                        value = df[col].iloc[0]
                        if value not in encoder.classes_:
                            return jsonify({
                                "error": f"Invalid value for '{col}': {value}. Expected one of: {list(encoder.classes_)}",
                                "status": "error"
                            }), 400
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        return jsonify({
                            "error": f"Invalid value for '{col}'. {str(e)}",
                            "status": "error"
                        }), 400
        
        # Apply scaling if scaler is available
        if scaler:
            try:
                df = pd.DataFrame(scaler.transform(df), columns=df.columns)
            except Exception as e:
                print(f"Warning: Could not apply scaling: {e}")
        
        # Make prediction
        prediction = model.predict(df)[0]
        
        # Get probability if available
        probability = None
        if hasattr(model, 'predict_proba'):
            probability = float(model.predict_proba(df)[0][1])
        
        # Prepare response with detailed risk categorization
        if prediction == 1:
            if probability and probability > 0.8:
                risk_label = "Very High Risk"
            elif probability and probability > 0.6:
                risk_label = "High Risk"
            else:
                risk_label = "Medium Risk"
        else:
            if probability and probability < 0.2:
                risk_label = "No Risk"
            elif probability and probability < 0.4:
                risk_label = "Low Risk"
            else:
                risk_label = "Low-Medium Risk"
        
        message = get_risk_message(prediction, probability)
        
        response = {
            "risk": int(prediction),
            "risk_label": risk_label,
            "status": "success"
        }
        
        if probability is not None:
            response["probability"] = round(probability, 4)
        
        response["message"] = message
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Predict harassment risk for multiple records.
    
    Expected JSON format:
    {
        "data": [
            {"age": 25, "occupation": "Student", ...},
            {"age": 30, "occupation": "Working", ...},
            ...
        ]
    }
    
    Returns:
    {
        "predictions": [
            {"risk": 1, "risk_label": "High Risk", "probability": 0.85},
            {"risk": 0, "risk_label": "Low Risk", "probability": 0.23},
            ...
        ],
        "status": "success"
    }
    """
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({
                "error": "Model not loaded. Please train the model first.",
                "status": "error"
            }), 500
        
        # Get JSON data from request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({
                "error": "No data provided. Please send JSON data with 'data' key.",
                "status": "error"
            }), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(request_data['data'])
        
        # Apply label encoding if encoders are available
        if label_encoders:
            for col, encoder in label_encoders.items():
                if col in df.columns:
                    try:
                        df[col] = encoder.transform(df[col])
                    except ValueError as e:
                        return jsonify({
                            "error": f"Invalid value for '{col}'. {str(e)}",
                            "status": "error"
                        }), 400
        
        # Apply scaling if scaler is available
        if scaler:
            try:
                df = pd.DataFrame(scaler.transform(df), columns=df.columns)
            except Exception as e:
                print(f"Warning: Could not apply scaling: {e}")
        
        # Make predictions
        predictions = model.predict(df)
        
        # Get probabilities if available
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(df)[:, 1]
        
        # Prepare response
        results = []
        for i, pred in enumerate(predictions):
            result = {
                "risk": int(pred),
                "risk_label": "High Risk" if pred == 1 else "Low Risk"
            }
            if probabilities is not None:
                result["probability"] = round(float(probabilities[i]), 4)
            results.append(result)
        
        return jsonify({
            "predictions": results,
            "count": len(results),
            "status": "success"
        }), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get information about the loaded model"""
    try:
        if model is None:
            return jsonify({
                "error": "Model not loaded.",
                "status": "error"
            }), 500
        
        info = {
            "model_type": type(model).__name__,
            "model_loaded": True,
            "scaler_loaded": scaler is not None,
            "encoders_loaded": label_encoders is not None,
            "status": "success"
        }
        
        # Try to get feature names
        try:
            if hasattr(model, 'feature_names_in_'):
                info['feature_names'] = model.feature_names_in_.tolist()
            elif hasattr(model, 'n_features_in_'):
                info['n_features'] = model.n_features_in_
        except:
            pass
        
        return jsonify(info), 200
    
    except Exception as e:
        return jsonify({
            "error": f"Failed to get model info: {str(e)}",
            "status": "error"
        }), 500

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Upload and process a CSV file with multiple questionnaire responses
    
    Expected CSV columns:
    - All the questionnaire columns matching the model's expected format
    
    Returns:
    {
        "status": "success",
        "total": 100,
        "processed": 98,
        "failed": 2,
        "results": [...]
    }
    """
    try:
        # Check if file is present in the request
        if 'file' not in request.files:
            return jsonify({
                "error": "No file part in the request",
                "status": "error"
            }), 400
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            return jsonify({
                "error": "No file selected",
                "status": "error"
            }), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return jsonify({
                "error": "Only CSV files are allowed",
                "status": "error"
            }), 400
        
        # Save the file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Read the CSV file
            df = pd.read_csv(filepath)
            total_records = len(df)
            
            if total_records == 0:
                return jsonify({
                    "error": "CSV file is empty",
                    "status": "error"
                }), 400
            
            # Process each record
            results = []
            processed = 0
            failed = 0
            
            for idx, row in df.iterrows():
                try:
                    # Convert row to dictionary
                    data = row.to_dict()
                    
                    # Create a single-row DataFrame for prediction
                    pred_df = pd.DataFrame([data])
                    
                    # Apply label encoding if encoders are available
                    if label_encoders:
                        for col, encoder in label_encoders.items():
                            if col == 'Timestamp':
                                continue
                            if col in pred_df.columns:
                                value = pred_df[col].iloc[0]
                                if pd.notna(value) and value in encoder.classes_:
                                    pred_df[col] = encoder.transform([value])
                                else:
                                    # Skip invalid values
                                    raise ValueError(f"Invalid value for column '{col}': {value}")
                    
                    # Apply scaling if scaler is available
                    if scaler:
                        pred_df = pd.DataFrame(scaler.transform(pred_df), columns=pred_df.columns)
                    
                    # Make prediction
                    prediction = model.predict(pred_df)[0]
                    
                    # Get probability if available
                    probability = None
                    if hasattr(model, 'predict_proba'):
                        probability = float(model.predict_proba(pred_df)[0][1])
                    
                    # Determine risk label
                    if prediction == 1:
                        if probability and probability > 0.8:
                            risk_label = "Very High Risk"
                        elif probability and probability > 0.6:
                            risk_label = "High Risk"
                        else:
                            risk_label = "Medium Risk"
                    else:
                        if probability and probability < 0.2:
                            risk_label = "No Risk"
                        elif probability and probability < 0.4:
                            risk_label = "Low Risk"
                        else:
                            risk_label = "Low-Medium Risk"
                    
                    results.append({
                        "row": idx + 1,
                        "risk": int(prediction),
                        "risk_label": risk_label,
                        "probability": round(probability, 4) if probability else None
                    })
                    
                    processed += 1
                    
                except Exception as e:
                    results.append({
                        "row": idx + 1,
                        "error": str(e)
                    })
                    failed += 1
            
            # Clean up - remove the uploaded file
            if os.path.exists(filepath):
                os.remove(filepath)
            
            return jsonify({
                "status": "success",
                "total": total_records,
                "processed": processed,
                "failed": failed,
                "results": results[:100]  # Return first 100 results
            }), 200
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                "error": f"Error processing CSV file: {str(e)}",
                "status": "error"
            }), 500
    
    except Exception as e:
        return jsonify({
            "error": f"Upload failed: {str(e)}",
            "status": "error"
        }), 500

@app.route('/api/statistics', methods=['GET'])
def get_statistics():
    """Get statistics from stored predictions"""
    try:
        # Try to load the CSV with predictions
        csv_path = "data/women_risk.csv"
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Get the risk column (last column)
            risk_column = df.columns[-1]
            
            # Count different risk levels
            total_responses = len(df)
            
            # Count based on risk level text
            no_risk = len(df[df[risk_column].str.contains('no risk', case=False, na=False)])
            low_risk = len(df[df[risk_column].str.contains('low risk', case=False, na=False) & 
                          ~df[risk_column].str.contains('no risk', case=False, na=False)])
            moderate_risk = len(df[df[risk_column].str.contains('moderate risk', case=False, na=False)])
            high_risk = len(df[df[risk_column].str.contains('high risk', case=False, na=False) & 
                           ~df[risk_column].str.contains('very high', case=False, na=False)])
            very_high_risk = len(df[df[risk_column].str.contains('very high risk', case=False, na=False)])
            
            # Combine moderate and high for display
            medium_risk = moderate_risk
            high_risk_total = high_risk + very_high_risk
            
            return jsonify({
                "status": "success",
                "total_responses": total_responses,
                "no_risk": no_risk,
                "low_risk": low_risk,
                "medium_risk": medium_risk,
                "high_risk": high_risk_total
            }), 200
        else:
            # Return zeros if no data yet
            return jsonify({
                "status": "success",
                "total_responses": 0,
                "no_risk": 0,
                "low_risk": 0,
                "medium_risk": 0,
                "high_risk": 0
            }), 200
            
    except Exception as e:
        return jsonify({
            "error": f"Failed to get statistics: {str(e)}",
            "status": "error"
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "Women Risk Predictor API is running"
    }), 200

@app.route('/notebooks/<notebook_name>')
def serve_notebook(notebook_name):
    """Serve notebook files"""
    notebook_path = os.path.join('scripts', f'{notebook_name}.ipynb')
    
    if not os.path.exists(notebook_path):
        return jsonify({
            "error": f"Notebook '{notebook_name}' not found",
            "status": "error"
        }), 404
    
    # Return info about opening the notebook
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>{notebook_name.replace('_', ' ').title()}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #fce4ec 0%, #f8bbd0 100%);
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                margin: 0;
                padding: 20px;
            }}
            .container {{
                background: white;
                padding: 40px;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 600px;
            }}
            h1 {{
                color: #e91e63;
                margin-bottom: 20px;
            }}
            p {{
                color: #666;
                line-height: 1.6;
                margin-bottom: 30px;
            }}
            .btn {{
                display: inline-block;
                padding: 12px 30px;
                background: #e91e63;
                color: white;
                text-decoration: none;
                border-radius: 8px;
                font-weight: bold;
                margin: 10px;
                transition: background 0.3s;
            }}
            .btn:hover {{
                background: #c2185b;
            }}
            .btn-secondary {{
                background: #2196f3;
            }}
            .btn-secondary:hover {{
                background: #1976d2;
            }}
            .path {{
                background: #f5f5f5;
                padding: 10px;
                border-radius: 5px;
                font-family: monospace;
                margin: 20px 0;
                word-break: break-all;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>📊 {notebook_name.replace('_', ' ').title()}</h1>
            <p>This notebook contains the implementation for {notebook_name.replace('_', ' ')}.</p>
            
            <div class="path">
                <strong>Notebook Path:</strong><br>
                backend/scripts/{notebook_name}.ipynb
            </div>
            
            <p>To open and run this notebook:</p>
            <ol style="text-align: left; color: #666; line-height: 2;">
                <li>Open Jupyter Notebook or JupyterLab</li>
                <li>Navigate to the backend/scripts folder</li>
                <li>Open {notebook_name}.ipynb</li>
                <li>Run the cells to execute the analysis</li>
            </ol>
            
            <a href="/dashboard" class="btn">← Back to Dashboard</a>
            <a href="javascript:history.back()" class="btn btn-secondary">Go Back</a>
        </div>
    </body>
    </html>
    """

def get_risk_message(prediction, probability=None):
    """Generate a message based on the risk prediction"""
    if prediction == 1:
        if probability and probability > 0.8:
            return "Very high risk detected. Please take immediate precautions and consider safer alternatives. Contact emergency services if needed."
        elif probability and probability > 0.6:
            return "High risk detected. Please be extremely cautious and take necessary safety measures. Inform someone about your location."
        else:
            return "Moderate-high risk detected. Stay alert and avoid risky situations. Consider alternative routes or timing."
    else:
        if probability and probability < 0.2:
            return "Very low risk detected. You are relatively safe in this situation. However, always remain aware of your surroundings."
        elif probability and probability < 0.4:
            return "Low risk detected. Continue to practice general safety measures and trust your instincts."
        else:
            return "Relatively low risk. Stay vigilant and follow basic safety protocols. Be prepared to take action if needed."

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        "error": "Endpoint not found",
        "status": "error"
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == "__main__":
    print("=" * 60)
    print("WOMEN RISK PREDICTOR API")
    print("=" * 60)
    print("\nStarting Flask server...")
    print("\nAvailable endpoints:")
    print("  GET  /              - API documentation")
    print("  GET  /dashboard     - Interactive dashboard")
    print("  POST /predict       - Single prediction")
    print("  POST /predict_batch - Batch prediction")
    print("  POST /upload        - Upload CSV file for batch prediction")
    print("  GET  /api/statistics- Get statistics")
    print("  GET  /model_info    - Model information")
    print("  GET  /notebooks/<name> - View notebook information")
    print("  GET  /health        - Health check")
    print("\nDashboard: http://127.0.0.1:5000/dashboard")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
