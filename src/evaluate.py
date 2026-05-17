import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from data.load_datasets import load_mnist_data, load_cifar10_data

def evaluate_model(model, dataset_name='mnist'):
    if dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist_data()
    else:
        (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    
    predictions = model.predict(x_test)
    predicted_classes = np.argmax(predictions, axis=1)
    
    test_labels = y_test.flatten()
    
    print(f"\n=== {dataset_name.upper()} Evaluation ===")
    print(classification_report(test_labels, predicted_classes))
    
    cm = confusion_matrix(test_labels, predicted_classes)
    plot_confusion_matrix(cm, dataset_name)
    
    visualize_predictions(x_test, test_labels, predicted_classes, dataset_name)
    
    return predicted_classes

def plot_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'{dataset_name.upper()} Confusion Matrix')
    plt.savefig(f'models/{dataset_name}_confusion_matrix.png')
    plt.close()
    print(f"Confusion matrix saved to models/{dataset_name}_confusion_matrix.png")

def visualize_predictions(x_test, true_labels, predicted_labels, dataset_name, num_samples=10):
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
    from tensorflow.keras.models import load_model
    
    try:
        model = load_model('models/mnist_final.h5')
        evaluate_model(model, 'mnist')
    except:
        print("MNIST model not found. Train first.")
    
    try:
        model = load_model('models/cifar10_final.h5')
        evaluate_model(model, 'cifar10')
    except:
        print("CIFAR-10 model not found. Train first.")
