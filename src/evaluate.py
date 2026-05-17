import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.load_datasets import load_mnist_data, load_cifar10_data

def evaluate_model(model, dataset_name='mnist', is_sklearn=True):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    else:
        (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    
    if is_sklearn:
        x_test_flat = x_test.reshape(x_test.shape[0], -1)
        predictions = model.predict(x_test_flat)
    else:
        predictions = model.predict(x_test)
        predictions = np.argmax(predictions, axis=1)
    
    test_labels = y_test.flatten()
    
    print(f"\n=== {dataset_name.upper()} Evaluation ===")
    print(classification_report(test_labels, predictions))
    
    cm = confusion_matrix(test_labels, predictions)
    plot_confusion_matrix(cm, dataset_name)
    
    visualize_predictions(x_test, test_labels, predictions, dataset_name)
    
    return predictions

def plot_confusion_matrix(cm, dataset_name):
    os.makedirs('models', exist_ok=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{dataset_name.upper()} Confusion Matrix')
    plt.savefig(f'models/{dataset_name}_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to models/{dataset_name}_confusion_matrix.png")

def visualize_predictions(x_test, true_labels, predicted_labels, dataset_name, num_samples=10):
    os.makedirs('models', exist_ok=True)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.flatten()
    
    incorrect = np.where(true_labels != predicted_labels)[0]
    
    if len(incorrect) >= num_samples:
        indices = np.random.choice(incorrect, num_samples, replace=False)
        title_prefix = "Incorrect: "
    else:
        indices = np.random.choice(len(x_test), num_samples, replace=False)
        title_prefix = ""
    
    for i, idx in enumerate(indices):
        axes[i].imshow(x_test[idx].squeeze(), cmap='gray' if len(x_test[idx].shape) < 3 else None)
        color = 'red' if true_labels[idx] != predicted_labels[idx] else 'green'
        axes[i].set_title(f"{title_prefix}True: {true_labels[idx]}\nPred: {predicted_labels[idx]}", color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'models/{dataset_name}_predictions.png')
    plt.close()
    print(f"Predictions visualization saved to models/{dataset_name}_predictions.png")

if __name__ == "__main__":
    from sklearn.ensemble import RandomForestClassifier
    from data.load_datasets import load_mnist_data
    
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    print("Training Random Forest...")
    model.fit(x_train_flat, y_train)
    
    evaluate_model(model, 'mnist', is_sklearn=True)
