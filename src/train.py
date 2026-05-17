import os
import joblib
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score
from src.models import create_svm_model

def train_mnist():
    print("=== Training SVM on MNIST ===")
    print("Downloading MNIST via OpenML (may take a moment)...")
    
    # Load data
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    x_all = mnist.data.astype('float32') / 255.0
    y_all = mnist.target.astype('int32')
    
    # Split
    x_train, x_test = x_all[:60000], x_all[60000:]
    y_train, y_test = y_all[:60000], y_all[60000:]
    
    # Create and train model
    model = create_svm_model()
    print("Training SVM (this may take 2-5 minutes)...")
    model.fit(x_train, y_train)
    
    # Evaluate
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    
    # Save
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/mnist_svm.pkl')
    print("Model saved to models/mnist_svm.pkl")
    
    return model

if __name__ == "__main__":
    train_mnist()
