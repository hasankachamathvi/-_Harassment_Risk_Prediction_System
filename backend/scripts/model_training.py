"""
Step 4-7: Model Training, Evaluation, Hyperparameter Tuning, and Saving
This script trains multiple models, evaluates them, performs hyperparameter tuning,
and saves the best model.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_processed_data(filepath):
    """Load the processed dataset"""
    print("=" * 60)
    print("LOADING PROCESSED DATASET")
    print("=" * 60)
    data = pd.read_csv(filepath)
    print(f"\nDataset loaded successfully!")
    print(f"Shape: {data.shape}")
    return data

def split_data(data, target_col='risk', test_size=0.2, random_state=42):
    """Split data into training and testing sets"""
    print("\n" + "=" * 60)
    print("SPLITTING DATA INTO TRAIN AND TEST SETS")
    print("=" * 60)
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nTraining set size: {X_train.shape[0]} ({(1-test_size)*100:.0f}%)")
    print(f"Testing set size: {X_test.shape[0]} ({test_size*100:.0f}%)")
    print(f"\nFeatures: {X_train.shape[1]}")
    print(f"\nTraining set target distribution:")
    print(y_train.value_counts())
    print(f"\nTesting set target distribution:")
    print(y_test.value_counts())
    
    return X_train, X_test, y_train, y_test

def train_multiple_models(X_train, y_train):
    """Train and compare multiple classification models"""
    print("\n" + "=" * 60)
    print("TRAINING MULTIPLE MODELS")
    print("=" * 60)
    
    # Define models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    results = {}
    
    print("\nPerforming 5-fold cross-validation for each model...\n")
    print(f"{'Model':<25} {'Mean Accuracy':<15} {'Std Dev':<10}")
    print("-" * 50)
    
    for name, model in models.items():
        # Cross-validation
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        results[name] = {
            'model': model,
            'mean_accuracy': scores.mean(),
            'std_dev': scores.std(),
            'scores': scores
        }
        print(f"{name:<25} {scores.mean():<15.4f} {scores.std():<10.4f}")
    
    # Visualize results
    plt.figure(figsize=(12, 6))
    model_names = list(results.keys())
    mean_accuracies = [results[name]['mean_accuracy'] for name in model_names]
    std_devs = [results[name]['std_dev'] for name in model_names]
    
    plt.bar(model_names, mean_accuracies, yerr=std_devs, capsize=5, 
            color='skyblue', edgecolor='navy', alpha=0.7)
    plt.xlabel('Model', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('Model Comparison (5-Fold Cross-Validation)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.ylim([min(mean_accuracies) - 0.05, 1.0])
    plt.tight_layout()
    plt.savefig('../data/model_comparison.png', dpi=300, bbox_inches='tight')
    print("\nModel comparison plot saved to '../data/model_comparison.png'")
    plt.close()
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['mean_accuracy'])
    print(f"\n{'='*50}")
    print(f"BEST MODEL: {best_model_name}")
    print(f"Accuracy: {results[best_model_name]['mean_accuracy']:.4f}")
    print(f"{'='*50}")
    
    return results, best_model_name

def evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    """Evaluate the model on test set"""
    print("\n" + "=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')
    
    print(f"\n{'Metric':<20} {'Score':<10}")
    print("-" * 30)
    print(f"{'Accuracy:':<20} {accuracy:<10.4f}")
    print(f"{'Precision:':<20} {precision:<10.4f}")
    print(f"{'Recall:':<20} {recall:<10.4f}")
    print(f"{'F1-Score:':<20} {f1:<10.4f}")
    
    if y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        print(f"{'ROC-AUC:':<20} {roc_auc:<10.4f}")
    
    # Classification Report
    print("\n" + "-" * 60)
    print("CLASSIFICATION REPORT")
    print("-" * 60)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Low Risk (0)', 'High Risk (1)'],
                yticklabels=['Low Risk (0)', 'High Risk (1)'])
    plt.title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'../data/confusion_matrix_{model_name.replace(" ", "_").lower()}.png', 
                dpi=300, bbox_inches='tight')
    print(f"\nConfusion matrix saved to '../data/confusion_matrix_{model_name.replace(' ', '_').lower()}.png'")
    plt.close()
    
    # ROC Curve
    if y_pred_proba is not None:
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'../data/roc_curve_{model_name.replace(" ", "_").lower()}.png', 
                    dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to '../data/roc_curve_{model_name.replace(' ', '_').lower()}.png'")
        plt.close()
    
    return model, accuracy, precision, recall, f1

def hyperparameter_tuning(model_name, X_train, y_train):
    """Perform hyperparameter tuning using GridSearchCV"""
    print("\n" + "=" * 60)
    print(f"HYPERPARAMETER TUNING FOR {model_name}")
    print("=" * 60)
    
    param_grids = {
        "Random Forest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10, 15],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "Gradient Boosting": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5, 7],
            "min_samples_split": [2, 5, 10]
        },
        "Logistic Regression": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ['l2'],
            "solver": ['lbfgs', 'liblinear']
        },
        "Decision Tree": {
            "max_depth": [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "criterion": ['gini', 'entropy']
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ['linear', 'rbf'],
            "gamma": ['scale', 'auto']
        }
    }
    
    models_dict = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }
    
    if model_name in param_grids:
        print(f"\nSearching best parameters for {model_name}...")
        print(f"Parameter grid: {param_grids[model_name]}")
        
        grid = GridSearchCV(
            models_dict[model_name], 
            param_grids[model_name], 
            cv=5, 
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid.fit(X_train, y_train)
        
        print("\n" + "-" * 60)
        print("BEST PARAMETERS FOUND:")
        print("-" * 60)
        for param, value in grid.best_params_.items():
            print(f"{param}: {value}")
        
        print(f"\nBest Cross-Validation Accuracy: {grid.best_score_:.4f}")
        
        return grid.best_estimator_, grid.best_params_
    else:
        print(f"\nNo hyperparameter grid defined for {model_name}")
        return models_dict[model_name], {}

def save_model(model, filepath):
    """Save the trained model"""
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)
    
    joblib.dump(model, filepath)
    print(f"\nModel saved to: {filepath}")

def save_model_info(model_name, params, metrics, filepath):
    """Save model information to a text file"""
    with open(filepath, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("WOMEN RISK PREDICTION MODEL INFORMATION\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: {model_name}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Best Hyperparameters:\n")
        f.write("-" * 60 + "\n")
        for param, value in params.items():
            f.write(f"{param}: {value}\n")
        f.write("\n" + "-" * 60 + "\n")
        f.write("Performance Metrics:\n")
        f.write("-" * 60 + "\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    print(f"Model information saved to: {filepath}")

def main():
    """Main function to execute model training pipeline"""
    # Define file paths
    input_file = "../data/women_risk_processed.csv"
    model_file = "../models/women_risk_model.pkl"
    model_info_file = "../models/model_info.txt"
    
    # Step 1: Load processed data
    data = load_processed_data(input_file)
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test = split_data(data)
    
    # Step 3: Train multiple models
    results, best_model_name = train_multiple_models(X_train, y_train)
    
    # Step 4: Evaluate best model (before tuning)
    print("\n" + "=" * 60)
    print("INITIAL EVALUATION (BEFORE HYPERPARAMETER TUNING)")
    print("=" * 60)
    best_model = results[best_model_name]['model']
    evaluate_model(best_model, X_train, X_test, y_train, y_test, best_model_name)
    
    # Step 5: Hyperparameter tuning
    tuned_model, best_params = hyperparameter_tuning(best_model_name, X_train, y_train)
    
    # Step 6: Evaluate tuned model
    print("\n" + "=" * 60)
    print("FINAL EVALUATION (AFTER HYPERPARAMETER TUNING)")
    print("=" * 60)
    final_model, accuracy, precision, recall, f1 = evaluate_model(
        tuned_model, X_train, X_test, y_train, y_test, best_model_name
    )
    
    # Step 7: Save the final model
    save_model(final_model, model_file)
    
    # Step 8: Save model information
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }
    save_model_info(best_model_name, best_params, metrics, model_info_file)
    
    print("\n" + "=" * 60)
    print("MODEL TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFinal Model: {best_model_name}")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Model saved to: {model_file}")
    
    return final_model

if __name__ == "__main__":
    main()
