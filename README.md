# Face the Data Training: Algorithms to Decode Human Emotions
This project explores the use of Convolutional Neural Networks (CNNs) to classify facial expressions into seven basic emotion categories using the FER2013 dataset. By building and evaluating several deep learning models, we aim to understand the effectiveness of different architectures in recognizing subtle human emotions from grayscale facial images.

---

## Data Preparation  

The dataset used for this project is [FER2013](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data), a widely used benchmark for emotion recognition.

- **Input size:** 48×48 pixels  
- **Color format:** Grayscale (single channel)  
- **Classes:**  
  - 0: Angry  
  - 1: Disgust  
  - 2: Fear  
  - 3: Happy  
  - 4: Sad  
  - 5: Surprise  
  - 6: Neutral  
- **Train/Test split:** Predefined by the dataset  
- **Preprocessing:**  
  - Normalization of pixel values to [0, 1]  
  - Data augmentation: small rotations, shifts, zooms, and horizontal flips

---

## Methodology  

We implemented and compared three neural network models:

### Baseline CNN  
A lightweight custom CNN model with:
- 3 convolutional layers with ReLU activations and max pooling  
- Fully connected layers with dropout regularization  
- Categorical crossentropy loss and Adam optimizer  

Achieved solid baseline performance and served as a control to evaluate deeper architectures.

### ResNet-Style Model  
Inspired by the ResNet architecture ([He et al., 2015](https://arxiv.org/abs/1512.03385)), this model includes:
- Residual blocks with skip connections  
- Improved gradient flow and training stability  
- Moderately deeper than baseline

Helped mitigate vanishing gradient issues and offered a performance boost over the CNN.

### VGG-Style Model  
A deeper and more expressive architecture modeled after VGG-16:
- Stacked 3×3 convolution filters with ReLU  
- Batch normalization and dropout  
- More parameters and layers

Despite its complexity, this model produced the highest test accuracy in our experiments.

---

## Performance Summary  

| Model          | Test Accuracy |
|----------------|---------------|
| Baseline CNN   | ~58%          |
| ResNet-Style   | ~60%          |
| VGG-Style      | ~62%          |

- **VGG-style model** delivered the best performance due to its depth and abstraction power.  
- **ResNet-style** improved stability but was not as expressive in our case.  
- The **Baseline CNN**, while simple, laid a good foundation.

---

## Key Takeaways  

- Facial expression recognition on low-resolution grayscale data remains a challenging task.  
- Model depth and architecture significantly affect accuracy.  
- Data augmentation and preprocessing were critical to avoid overfitting.  
- Transfer learning or higher-resolution datasets may offer future improvements.  

---

## Model Deployment  

This repository includes trained model files and a sample Jupyter notebook for inference.  
To test with your own images or run a demo, see the `model_inference_demo.ipynb` file (if included in the repo).

---

## References  

- He, K., Zhang, X., Ren, S., & Sun, J. (2015). [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). *arXiv preprint arXiv:1512.03385*.
- We used the [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) available on Kaggle.
Goodfellow, I., Erhan, D., Luc Carrier, Courville, A., & Bengio, Y. (2013).  
Challenges in Representation Learning: A report on three machine learning contests.  
Neural Information Processing Systems (NIPS), Workshop on Deep Learning and Unsupervised Feature Learning.
Dataset hosted on: https://www.kaggle.com/datasets/msambare/fer2013 (download the zipfile in order to use on local machine)

---

