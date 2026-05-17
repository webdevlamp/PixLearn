import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import numpy as np
import joblib
from PIL import Image

MODEL_PATH = 'models/mnist_svm.pkl'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run src/train.py first.")

model = joblib.load(MODEL_PATH)

def predict_digit(image):
    if image is None:
        return "Please upload an image.", 0.0
    
    # 1. Grayscale and Resize
    img = image.convert('L')
    img = img.resize((28, 28))
    img_array = np.array(img).astype('float32')
    
    # 2. Invert colors if necessary (MNIST is white on black)
    if np.mean(img_array) > 127:
        img_array = 255 - img_array
    
    # 3. Normalize
    img_array = img_array / 255.0
    
    # 4. Center the digit (MNIST style preprocessing)
    coords = np.column_stack(np.where(img_array > 0.1))
    if len(coords) > 0:
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        digit_roi = img_array[y_min:y_max+1, x_min:x_max+1]
        digit_roi = Image.fromarray((digit_roi * 255).astype('uint8'))
        digit_roi = digit_roi.resize((20, 20))
        digit_roi = np.array(digit_roi).astype('float32') / 255.0
        
        new_img = np.zeros((28, 28), dtype='float32')
        new_img[4:24, 4:24] = digit_roi
        img_array = new_img
    
    img_array = img_array.reshape(1, -1)
    
    # Predict
    prediction = model.predict(img_array)[0]
    probabilities = model.predict_proba(img_array)[0]
    confidence = probabilities[prediction]
    
    return f"Predicted Digit: {prediction}", f"Confidence: {confidence:.2%}"

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload Handwritten Digit"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="PixLearn - Digit Recognizer (SVM)",
    description="Upload an image of a handwritten digit (0-9) to classify it.",
)

if __name__ == "__main__":
    iface.launch()
