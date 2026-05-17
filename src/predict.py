import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import joblib
from PIL import Image

def predict_image(image_path, model_path='models/mnist_rf_model.pkl'):
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_sklearn.py first.")
        return
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    print(f"Loading image: {image_path}")
    try:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        img = img.resize((28, 28))  # Resize to MNIST size
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype('float32') / 255.0
        img_array = img_array.reshape(1, -1)  # Flatten for sklearn
        
        prediction = model.predict(img_array)[0]
        probabilities = model.predict_proba(img_array)[0]
        
        print(f"\nPrediction: {prediction}")
        print(f"Confidence: {probabilities[prediction]:.2%}")
        
        # Show top 3 guesses
        top_indices = np.argsort(probabilities)[::-1][:3]
        print("\nTop 3 guesses:")
        for idx in top_indices:
            print(f"  Digit {idx}: {probabilities[idx]:.2%}")
            
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict_image(sys.argv[1])
    else:
        print("Usage: python3 src/predict.py <path_to_image>")
        print("Example: python3 src/predict.py my_digit.png")
