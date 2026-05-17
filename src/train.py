import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from data.load_datasets import load_mnist_data, load_cifar10_data
from src.models import create_cnn_mnist, create_cnn_cifar10

def train_mnist(epochs=10, batch_size=64):
    print("=== Training MNIST CNN ===")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    
    model = create_cnn_mnist()
    
    callbacks = [
        EarlyStopping(patience=3, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2),
        ModelCheckpoint('models/mnist_best.h5', save_best_only=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nMNIST Test Accuracy: {test_acc:.4f}")
    
    model.save('models/mnist_final.h5')
    plot_training_history(history, 'mnist')
    
    return model, history

def train_cifar10(epochs=20, batch_size=64):
    print("=== Training CIFAR-10 CNN ===")
    (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    
    model = create_cnn_cifar10()
    
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=2),
        ModelCheckpoint('models/cifar10_best.h5', save_best_only=True)
    ]
    
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=callbacks
    )
    
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\nCIFAR-10 Test Accuracy: {test_acc:.4f}")
    
    model.save('models/cifar10_final.h5')
    plot_training_history(history, 'cifar10')
    
    return model, history

def plot_training_history(history, dataset_name):
    os.makedirs('models', exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(history.history['accuracy'], label='Train Accuracy')
    axes[0].plot(history.history['val_accuracy'], label='Val Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title(f'{dataset_name.upper()} Accuracy')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history.history['loss'], label='Train Loss')
    axes[1].plot(history.history['val_loss'], label='Val Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title(f'{dataset_name.upper()} Loss')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'models/{dataset_name}_training_history.png')
    plt.close()
    print(f"Training history saved to models/{dataset_name}_training_history.png")

if __name__ == "__main__":
    os.makedirs('models', exist_ok=True)
    
    train_mnist(epochs=10)
    train_cifar10(epochs=20)
