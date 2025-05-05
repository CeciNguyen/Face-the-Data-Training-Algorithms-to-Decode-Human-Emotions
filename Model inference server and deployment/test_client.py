import requests
import json
import sys
import os
from PIL import Image
import io

def test_summary_endpoint(base_url="http://localhost:5000"):
    """Test the GET /summary endpoint"""
    url = f"{base_url}/summary"
    
    try:
        response = requests.get(url)
        
        # Check status code
        if response.status_code == 200:
            print("GET /summary - Success!")
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"GET /summary - Failed with status code: {response.status_code}")
            print("Response:", response.text)
            return False
    
    except Exception as e:
        print(f"GET /summary - Error: {e}")
        return False

def test_inference_endpoint(image_path, base_url="http://localhost:5000"):
    """Test the POST /inference endpoint"""
    url = f"{base_url}/inference"
    
    try:
        # Check if the image exists
        if not os.path.exists(image_path):
            print(f"Image not found at: {image_path}")
            return False
        
        # Open the image file
        with open(image_path, "rb") as f:
            image_data = f.read()
        
        # Send POST request with binary image data
        headers = {"Content-Type": "application/octet-stream"}
        response = requests.post(url, data=image_data, headers=headers)
        
        # Check status code
        if response.status_code == 200:
            print("POST /inference - Success!")
            print("Response:")
            print(json.dumps(response.json(), indent=2))
            
            # Check response format
            resp_json = response.json()
            if "prediction" in resp_json:
                valid_emotions = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
                if resp_json["prediction"] in valid_emotions:
                    print("Response format contains a valid emotion prediction")
                else:
                    print(f"'prediction' value must be one of: {', '.join(valid_emotions)}")
            else:
                print("Response is missing 'prediction' field")
            
            return True
        else:
            print(f"POST /inference - Failed with status code: {response.status_code}")
            print("Response:", response.text)
            return False
    
    except Exception as e:
        print(f"POST /inference - Error: {e}")
        return False

if __name__ == "__main__":
    # Get base URL from command line argument or use default
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
    
    # Test summary endpoint
    test_summary_endpoint(base_url)
    
    # Test inference endpoint if image path is provided
    if len(sys.argv) > 2:
        image_path = sys.argv[2]
        test_inference_endpoint(image_path, base_url)
    else:
        print("\nTo test the inference endpoint, provide an image path:")
        print(f"python {sys.argv[0]} [base_url] [image_path]")