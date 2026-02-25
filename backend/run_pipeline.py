"""
Complete ML Pipeline Runner
Runs all steps from data preparation to model training in sequence.
"""

import os
import sys
import subprocess
import time

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def run_script(script_name, description):
    """Run a Python script and handle errors"""
    print_header(description)
    
    script_path = os.path.join("scripts", script_name)
    
    if not os.path.exists(script_path):
        print(f"❌ Error: Script '{script_path}' not found!")
        return False
    
    try:
        print(f"▶️  Running {script_name}...")
        print("-" * 70)
        
        # Change to backend directory if not already there
        original_dir = os.getcwd()
        if os.path.basename(original_dir) != "backend":
            backend_dir = os.path.join(os.path.dirname(__file__))
            os.chdir(backend_dir)
        
        # Run the script
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=False,
            text=True
        )
        
        os.chdir(original_dir)
        
        if result.returncode == 0:
            print("\n✅ Successfully completed!")
            return True
        else:
            print(f"\n❌ Script failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running script: {str(e)}")
        return False

def check_dataset():
    """Check if dataset exists"""
    print_header("Checking Dataset")
    
    dataset_path = "data/women_risk.csv"
    
    if os.path.exists(dataset_path):
        print(f"✅ Dataset found at: {dataset_path}")
        
        # Get file size
        size = os.path.getsize(dataset_path)
        size_mb = size / (1024 * 1024)
        print(f"📊 Dataset size: {size_mb:.2f} MB")
        
        return True
    else:
        print(f"❌ Dataset not found at: {dataset_path}")
        print("\nPlease ensure the dataset is placed at:")
        print(f"  {os.path.abspath(dataset_path)}")
        return False

def main():
    """Main function to run the complete ML pipeline"""
    start_time = time.time()
    
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 20 + "WOMEN RISK PREDICTOR" + " " * 28 + "║")
    print("║" + " " * 18 + "Complete ML Pipeline Runner" + " " * 23 + "║")
    print("╚" + "═" * 68 + "╝")
    
    # Step 0: Check dataset
    if not check_dataset():
        print("\n❌ Pipeline aborted: Dataset not found!")
        return
    
    # Pipeline steps
    steps = [
        ("data_preparation.py", "Step 1: Data Preparation & Cleaning"),
        ("feature_engineering.py", "Step 2: Feature Engineering"),
        ("model_training.py", "Step 3: Model Training & Evaluation")
    ]
    
    # Run each step
    results = []
    for script, description in steps:
        success = run_script(script, description)
        results.append((description, success))
        
        if not success:
            print("\n❌ Pipeline stopped due to error!")
            break
        
        time.sleep(1)  # Brief pause between steps
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print_header("Pipeline Summary")
    
    print("Step Results:")
    print("-" * 70)
    for step, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status:12} | {step}")
    
    print("-" * 70)
    
    total_steps = len(steps)
    passed_steps = sum(1 for _, success in results if success)
    
    print(f"\nTotal Steps: {total_steps}")
    print(f"Passed: {passed_steps}")
    print(f"Failed: {total_steps - passed_steps}")
    print(f"\nTotal Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
    
    if passed_steps == total_steps:
        print("\n" + "=" * 70)
        print("🎉 SUCCESS! All pipeline steps completed successfully!")
        print("=" * 70)
        print("\n📁 Generated files:")
        print("   ✓ data/women_risk_cleaned.csv")
        print("   ✓ data/women_risk_processed.csv")
        print("   ✓ models/women_risk_model.pkl")
        print("   ✓ models/scaler.pkl")
        print("   ✓ models/label_encoders.pkl")
        print("   ✓ models/model_info.txt")
        print("   ✓ Visualizations in data/ folder")
        
        print("\n🚀 Next Steps:")
        print("   1. Review the model performance metrics")
        print("   2. Check the generated visualizations")
        print("   3. Start the Flask API:")
        print("      python app.py")
        print("   4. Test predictions at http://127.0.0.1:5000")
        
    else:
        print("\n❌ Pipeline completed with errors!")
        print("Please check the error messages above and fix the issues.")
    
    print("\n")

if __name__ == "__main__":
    # Change to backend directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    main()
