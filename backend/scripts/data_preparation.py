"""
Step 2: Data Collection and Preparation
This script loads, explores, and cleans the women risk dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load the dataset from CSV file"""
    print("=" * 60)
    print("LOADING DATASET")
    print("=" * 60)
    data = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {data.shape}")
    return data

def explore_data(data):
    """Explore the dataset"""
    print("\n" + "=" * 60)
    print("DATA EXPLORATION")
    print("=" * 60)
    
    print("\n--- First 5 rows ---")
    print(data.head())
    
    print("\n--- Dataset Info ---")
    print(data.info())
    
    print("\n--- Statistical Summary ---")
    print(data.describe())
    
    print("\n--- Column Names ---")
    print(data.columns.tolist())
    
    return data

def check_missing_data(data):
    """Check for missing values"""
    print("\n" + "=" * 60)
    print("CHECKING MISSING DATA")
    print("=" * 60)
    
    missing = data.isnull().sum()
    print("\nMissing values per column:")
    print(missing)
    
    if missing.sum() > 0:
        print(f"\nTotal missing values: {missing.sum()}")
        print("\nDropping rows with missing values...")
        data = data.dropna()
        print(f"New shape: {data.shape}")
    else:
        print("\nNo missing values found!")
    
    return data

def remove_duplicates(data):
    """Remove duplicate rows"""
    print("\n" + "=" * 60)
    print("REMOVING DUPLICATES")
    print("=" * 60)
    
    initial_rows = len(data)
    data = data.drop_duplicates()
    final_rows = len(data)
    
    duplicates_removed = initial_rows - final_rows
    print(f"\nDuplicates removed: {duplicates_removed}")
    print(f"Final shape: {data.shape}")
    
    return data

def encode_categorical_variables(data):
    """Encode categorical variables using Label Encoding"""
    print("\n" + "=" * 60)
    print("ENCODING CATEGORICAL VARIABLES")
    print("=" * 60)
    
    # Identify categorical columns
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        print(f"\nCategorical columns found: {categorical_cols}")
        
        le = LabelEncoder()
        label_encoders = {}
        
        for col in categorical_cols:
            print(f"\nEncoding '{col}'...")
            print(f"Unique values before encoding: {data[col].nunique()}")
            data[col] = le.fit_transform(data[col])
            label_encoders[col] = le
            print(f"Encoding completed for '{col}'")
        
        # Save label encoders for later use
        import joblib
        joblib.dump(label_encoders, 'models/label_encoders.pkl')
        print("\nLabel encoders saved to 'models/label_encoders.pkl'")
    else:
        print("\nNo categorical columns found!")
    
    return data

def save_cleaned_data(data, filepath):
    """Save the cleaned dataset"""
    print("\n" + "=" * 60)
    print("SAVING CLEANED DATA")
    print("=" * 60)
    
    data.to_csv(filepath, index=False)
    print(f"\nCleaned data saved to: {filepath}")
    print(f"Final shape: {data.shape}")

def main():
    """Main function to execute data preparation pipeline"""
    # Define file paths
    input_file = "data/women_risk.csv"
    output_file = "data/women_risk_cleaned.csv"
    
    # Step 1: Load data
    data = load_data(input_file)
    
    # Step 2: Explore data
    data = explore_data(data)
    
    # Step 3: Check for missing data
    data = check_missing_data(data)
    
    # Step 4: Remove duplicates
    data = remove_duplicates(data)
    
    # Step 5: Encode categorical variables
    data = encode_categorical_variables(data)
    
    # Step 6: Save cleaned data
    save_cleaned_data(data, output_file)
    
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return data

if __name__ == "__main__":
    main()
