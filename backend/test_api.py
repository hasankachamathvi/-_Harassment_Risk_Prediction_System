"""
API Testing Script
Tests all endpoints of the Women Risk Predictor API
"""

import requests
import json
import sys

# Base URL for the API
BASE_URL = "http://127.0.0.1:5000"

def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")

def print_response(response):
    """Print formatted response"""
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))

def test_health():
    """Test the health endpoint"""
    print_header("Testing: GET /health")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response(response)
        return response.status_code == 200
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API!")
        print("Please make sure the Flask server is running:")
        print("  python app.py")
        return False
    except Exception as e:
        print(f" Error: {str(e)}")
        return False

def test_model_info():
    """Test the model info endpoint"""
    print_header("Testing: GET /model_info")
    
    try:
        response = requests.get(f"{BASE_URL}/model_info")
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {str(e)}")
        return False

def test_single_prediction():
    """Test single prediction endpoint"""
    print_header("Testing: POST /predict (Single Prediction)")
    
    # Sample data - adjust based on your actual dataset columns
    test_data = {
        "age": 25,
        "occupation": 0,  # Encoded value
        "location": 1,    # Encoded value
        "time_of_day": 3, # Encoded value
        "public_transport_usage": 1,
        "past_incidents": 2
    }
    
    print("Request Data:")
    print(json.dumps(test_data, indent=2))
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {str(e)}")
        return False

def test_batch_prediction():
    """Test batch prediction endpoint"""
    print_header("Testing: POST /predict_batch (Batch Prediction)")
    
    # Sample batch data
    test_data = {
        "data": [
            {
                "age": 25,
                "occupation": 0,
                "location": 1,
                "time_of_day": 3,
                "public_transport_usage": 1,
                "past_incidents": 2
            },
            {
                "age": 30,
                "occupation": 1,
                "location": 0,
                "time_of_day": 1,
                "public_transport_usage": 0,
                "past_incidents": 0
            },
            {
                "age": 22,
                "occupation": 0,
                "location": 1,
                "time_of_day": 2,
                "public_transport_usage": 1,
                "past_incidents": 1
            }
        ]
    }
    
    print("Request Data (3 records):")
    print(json.dumps(test_data, indent=2))
    print()
    
    try:
        response = requests.post(
            f"{BASE_URL}/predict_batch",
            json=test_data,
            headers={"Content-Type": "application/json"}
        )
        print_response(response)
        return response.status_code == 200
    except Exception as e:
        print(f" Error: {str(e)}")
        return False

def test_invalid_endpoint():
    """Test an invalid endpoint (should return 404)"""
    print_header("Testing: GET /invalid (Should Return 404)")
    
    try:
        response = requests.get(f"{BASE_URL}/invalid")
        print_response(response)
        return response.status_code == 404
    except Exception as e:
        print(f" Error: {str(e)}")
        return False

def main():
    """Main function to run all tests"""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "WOMEN RISK PREDICTOR API TESTING" + " " * 21 + "║")
    print("╚" + "═" * 68 + "╝")
    
    print(f"\n API Base URL: {BASE_URL}")
    
    # Run tests
    tests = [
        ("Health Check", test_health),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Invalid Endpoint", test_invalid_endpoint)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f" Test failed with exception: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print_header("Test Summary")
    
    print(f"{'Test Name':<30} {'Result':<15}")
    print("-" * 70)
    
    for test_name, success in results:
        status = " PASSED" if success else " FAILED"
        print(f"{test_name:<30} {status:<15}")
    
    print("-" * 70)
    
    total_tests = len(tests)
    passed_tests = sum(1 for _, success in results if success)
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n All tests passed!")
    else:
        print("\n Some tests failed. Please check the output above.")
    
    print("\n")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
