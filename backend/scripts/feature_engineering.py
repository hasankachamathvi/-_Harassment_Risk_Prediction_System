"""
Step 3: Feature Engineering
This script performs feature selection, correlation analysis, and feature scaling.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_cleaned_data(filepath):
    """Load the cleaned dataset"""
    print("=" * 60)
    print("LOADING CLEANED DATASET")
    print("=" * 60)
    data = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {data.shape}")
    return data

def correlation_analysis(data, target_col='risk'):
    """Perform correlation analysis and visualize"""
    print("\n" + "=" * 60)
    print("CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Calculate correlation matrix
    corr_matrix = data.corr()
    
    print("\nCorrelation with target variable (risk):")
    if target_col in corr_matrix.columns:
        target_corr = corr_matrix[target_col].sort_values(ascending=False)
        print(target_corr)
        
        # Visualize correlation heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                    fmt='.2f', square=True, linewidths=1)
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('data/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("\nCorrelation heatmap saved to 'data/correlation_heatmap.png'")
        plt.close()
        
        # Bar plot of correlations with target
        plt.figure(figsize=(10, 6))
        target_corr_filtered = target_corr[target_corr.index != target_col]
        target_corr_filtered.plot(kind='barh', color='steelblue')
        plt.title(f'Feature Correlation with {target_col}', fontsize=14, fontweight='bold')
        plt.xlabel('Correlation Coefficient')
        plt.tight_layout()
        plt.savefig('data/target_correlation.png', dpi=300, bbox_inches='tight')
        print("Target correlation plot saved to 'data/target_correlation.png'")
        plt.close()
    else:
        print(f"\nWarning: Target column '{target_col}' not found in dataset!")
        print(f"Available columns: {data.columns.tolist()}")
    
    return data

def create_new_features(data):
    """Create new features if beneficial"""
    print("\n" + "=" * 60)
    print("CREATING NEW FEATURES")
    print("=" * 60)
    
    initial_features = data.shape[1]
    
    # Create binary risk target variable from the last column
    # The last column should be the encoded risk level
    last_col = data.columns[-1]
    print(f"\nCreating binary 'risk' target from column: '{last_col}'")
    print(f"Unique values: {sorted(data[last_col].unique())}")
    
    # Create binary risk: 1 for high/moderate risk, 0 for low/no risk
    # Adjust threshold based on encoded values (usually median or specific value)
    median_value = data[last_col].median()
    data['risk'] = (data[last_col] >= median_value).astype(int)
    
    print(f"Binary risk distribution:")
    print(data['risk'].value_counts())
    print(f"Risk percentage: {(data['risk'].sum() / len(data) * 100):.2f}%")
    
    # Example: Create interaction features if columns exist
    # You can customize this based on your actual dataset columns
    
    # Check for common columns that might exist
    if 'age' in data.columns and 'past_incidents' in data.columns:
        data['risk_score'] = data['age'] * data['past_incidents']
        print("\nCreated 'risk_score' = age * past_incidents")
    
    if 'public_transport_usage' in data.columns and 'time_of_day' in data.columns:
        data['transport_time_interaction'] = data['public_transport_usage'] * data['time_of_day']
        print("Created 'transport_time_interaction' = public_transport_usage * time_of_day")
    
    final_features = data.shape[1]
    new_features = final_features - initial_features
    
    print(f"\nNew features created: {new_features}")
    print(f"Total features now: {final_features}")
    
    return data

def scale_numeric_features(data, target_col='risk'):
    """Scale numeric features using StandardScaler"""
    print("\n" + "=" * 60)
    print("SCALING NUMERIC FEATURES")
    print("=" * 60)
    
    # Identify numeric columns (excluding target)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    if numeric_cols:
        print(f"\nNumeric columns to scale: {numeric_cols}")
        
        scaler = StandardScaler()
        data_scaled = data.copy()
        data_scaled[numeric_cols] = scaler.fit_transform(data[numeric_cols])
        
        # Save the scaler
        joblib.dump(scaler, 'models/scaler.pkl')
        print("\nScaler saved to 'models/scaler.pkl'")
        
        # Show before and after statistics
        print("\n--- Statistics Before Scaling ---")
        print(data[numeric_cols].describe())
        
        print("\n--- Statistics After Scaling ---")
        print(data_scaled[numeric_cols].describe())
        
        return data_scaled
    else:
        print("\nNo numeric columns found to scale!")
        return data

def split_features_target(data, target_col='risk'):
    """Split data into features (X) and target (y)"""
    print("\n" + "=" * 60)
    print("SPLITTING FEATURES AND TARGET")
    print("=" * 60)
    
    if target_col in data.columns:
        X = data.drop(target_col, axis=1)
        y = data[target_col]
        
        print(f"\nFeatures shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"\nTarget distribution:")
        print(y.value_counts())
        print(f"\nTarget distribution (%):")
        print(y.value_counts(normalize=True) * 100)
        
        return X, y
    else:
        print(f"\nError: Target column '{target_col}' not found!")
        print(f"Available columns: {data.columns.tolist()}")
        return None, None

def save_processed_data(data, filepath):
    """Save the feature-engineered dataset"""
    print("\n" + "=" * 60)
    print("SAVING PROCESSED DATA")
    print("=" * 60)
    
    data.to_csv(filepath, index=False)
    print(f"\nProcessed data saved to: {filepath}")
    print(f"Shape: {data.shape}")

def main():
    """Main function to execute feature engineering pipeline"""
    # Define file paths
    input_file = "data/women_risk_cleaned.csv"
    output_file = "data/women_risk_processed.csv"
    
    # Step 1: Load cleaned data
    data = load_cleaned_data(input_file)
    
    # Step 2: Correlation analysis
    data = correlation_analysis(data)
    
    # Step 3: Create new features
    data = create_new_features(data)
    
    # Step 4: Scale numeric features
    data = scale_numeric_features(data)
    
    # Step 5: Split features and target (for verification)
    X, y = split_features_target(data)
    
    # Step 6: Save processed data
    save_processed_data(data, output_file)
    
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return data

if __name__ == "__main__":
    main()
