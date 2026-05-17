import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from data.load_datasets import load_mnist_data, load_cifar10_data
from src.models import create_sklearn_classifier

def train_sklearn_mnist(model_type='random_forest'):
    print(f"=== Training Scikit-Learn {model_type.upper()} on MNIST ===")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    model = create_sklearn_classifier(model_type)
    
    print(f"Training {model_type}...")
    model.fit(x_train_flat, y_train)
    
    y_pred = model.predict(x_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nMNIST Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    
    return model, y_pred

def train_sklearn_cifar10(model_type='random_forest', sample_size=5000):
    print(f"=== Training Scikit-Learn {model_type.upper()} on CIFAR-10 ===")
    (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    
    x_train = x_train[:sample_size]
    y_train = y_train[:sample_size]
    x_test = x_test[:sample_size]
    y_test = y_test[:sample_size]
    
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    model = create_sklearn_classifier(model_type)
    
    print(f"Training {model_type} on {sample_size} samples...")
    model.fit(x_train_flat, y_train)
    
    y_pred = model.predict(x_test_flat)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nCIFAR-10 Test Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, target_names=classes))
    
    return model, y_pred

if __name__ == "__main__":
    train_sklearn_mnist('random_forest')
