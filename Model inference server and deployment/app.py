from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import json
import datetime
import time

app = Flask(__name__)

# Model loading function
def load_model():
    """Load the trained emotion classification model from disk"""
    model_path = "emotion_model.h5"
    
    # Check if model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    try:
        # Try to load the model without optimizer
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully without optimizer")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

# Model metadata
MODEL_INFO = {
    "name": "Emotion Classifier",
    "version": "1.0.0",
    "framework": "TensorFlow/Keras",
    "precision": "FP32",
    "input_shape": [1, 48, 48, 1],  # Standard input size for FER2013
    "output_shape": [1, 7],
    "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
    "created_at": datetime.datetime.now().isoformat(),
    "accuracy": 0.6,  # Accuracy for your FER2013 model
    "dataset": "FER2013 (Facial Expression Recognition)"
}

# Preprocess image function
def preprocess_image(image_data):
    """
    Preprocess the image for the FER2013 model:
    - Convert to grayscale
    - Resize to 48x48
    - Normalize pixel values
    """
    # Open image from binary data
    img = Image.open(io.BytesIO(image_data))
    
    # Convert to grayscale
    if img.mode != 'L':
        img = img.convert('L')
    
    # Resize to 48x48 (standard size for FER2013)
    img = img.resize((48, 48))
    
    # Convert to numpy array and normalize
    img_array = np.array(img)
    img_array = img_array / 255.0  # Normalize to [0,1]
    
    # Reshape to model input shape [batch, height, width, channels]
    img_array = img_array.reshape(1, 48, 48, 1)
    
    return img_array

# Global variable to hold the model
global_model = None

@app.before_request
def initialize():
    """Initialize the model if it's not already loaded"""
    global global_model
    if global_model is None:
        global_model = load_model()

@app.route('/summary', methods=['GET'])
def get_summary():
    """Endpoint to get model metadata"""
    return jsonify(MODEL_INFO)

@app.route('/inference', methods=['POST'])
def inference():
    """Endpoint to perform inference on an image"""
    global global_model
    
    if global_model is None:
        try:
            global_model = load_model()
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    # Get the binary image data from the request
    if not request.data:
        return jsonify({"error": "No image data provided"}), 400
    
    try:
        # Preprocess the image
        processed_image = preprocess_image(request.data)
        
        # Perform inference
        predictions = global_model.predict(processed_image, verbose=0)
        
        # Get the predicted emotion class index
        emotion_idx = np.argmax(predictions[0])
        
        # Get the emotion label
        emotion = MODEL_INFO["classes"][emotion_idx]
        
        # Return a JSON response with the prediction
        return jsonify({"prediction": emotion})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load the model at startup
    global_model = load_model()
    
    # Run the server
    app.run(host='0.0.0.0', port=5000, debug=False)
