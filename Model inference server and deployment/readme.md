# Emotion Classification Server

A containerized HTTP server for facial emotion classification using FER2013 dataset, TensorFlow/Keras, and Flask.

## Overview

This project implements a web service that performs facial emotion classification. The server provides two RESTful API endpoints:
- A summary endpoint that returns model metadata
- An inference endpoint that accepts image data and returns the predicted emotion

The model classifies facial expressions into 7 categories: angry, disgust, fear, happy, sad, surprise, and neutral.

## Quick Start with Docker

The easiest way to run this server is using the pre-built Docker image:

```bash
# Pull the image from Docker Hub
docker pull ssspro/emotion-classifier:latest

# Run the container
docker run -p 8080:5000 ssspro/emotion-classifier:latest
```

The server will be available at http://localhost:8080

## API Documentation

### GET /summary

Returns metadata about the model as JSON.

**Example Request:**
```bash
curl http://localhost:8080/summary
```

**Example Response:**
```json
{
  "accuracy": 0.6,
  "classes": ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"],
  "created_at": "2025-05-05T18:12:49.992031",
  "dataset": "FER2013 (Facial Expression Recognition)",
  "framework": "TensorFlow/Keras",
  "input_shape": [1, 48, 48, 1],
  "name": "Emotion Classifier",
  "output_shape": [1, 7],
  "precision": "FP32",
  "version": "1.0.0"
}
```

### POST /inference

Accepts a binary image and returns the predicted emotion as JSON.

**Example Request:**
```bash
curl -X POST -H "Content-Type: application/octet-stream" \
  --data-binary "@path/to/face_image.jpg" \
  http://localhost:8080/inference
```

**Example Response:**
```json
{
  "prediction": "happy"
}
```

## Implementation Details

### Technology Stack

- **TensorFlow/Keras**: Deep learning framework for the emotion classification model
- **Flask**: Web framework for the RESTful API
- **Docker**: Container platform for packaging and deployment
- **PIL/Pillow**: Image processing library

### Model Architecture

The emotion classification model is trained on the FER2013 dataset, which contains 48x48 pixel grayscale images of faces labeled with 7 emotion categories. The model input shape is [1, 48, 48, 1] and output shape is [1, 7].

### Image Preprocessing

When an image is submitted for inference, the following preprocessing is applied:
1. Conversion to grayscale (if not already)
2. Resizing to 48x48 pixels (FER2013 standard size)
3. Normalization of pixel values to range [0, 1]
4. Reshaping to match model input requirements

## Local Development

### Prerequisites

- Python 3.8+
- TensorFlow 2.9.1
- Flask 2.0.1
- Pillow
- NumPy

### Setup

1. Clone the repository or download the source files

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure your trained model is named `emotion_model.h5` and placed in the project directory

4. Run the server:
   ```bash
   python app.py
   ```

### Build Custom Docker Image

```bash
# Build the image
docker build -t emotion-classifier .

# Run the container
docker run -p 8080:5000 emotion-classifier
```

## Testing

The repository includes a `test_client.py` script to test the API:

```bash
# Install test dependencies
pip install requests Pillow

# Test the summary endpoint
python test_client.py http://localhost:8080

# Test the inference endpoint with an image
python test_client.py http://localhost:8080 path/to/face_image.jpg
```

## License

This project is provided for educational purposes only.