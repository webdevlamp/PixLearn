import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from src.models import CnnModel

MODEL_PATH = 'models/mnist_cnn.pth'
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run src/train.py first.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CnnModel().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def predict_digit(image):
    if image is None:
        return "Please upload an image.", 0.0
    
    img = image.convert('L')
    
    # Invert if necessary (MNIST is white on black)
    if torch.tensor(list(img.getdata())).mean() > 127:
        img = Image.eval(img, lambda x: 255 - x)
        
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return f"Predicted Digit: {predicted.item()}", f"Confidence: {confidence.item():.2%}"

iface = gr.Interface(
    fn=predict_digit,
    inputs=gr.Image(type="pil", label="Upload Handwritten Digit"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence")
    ],
    title="PixLearn - Digit Recognizer (PyTorch CNN)",
    description="Upload an image of a handwritten digit (0-9) to classify it.",
)

if __name__ == "__main__":
    iface.launch()
