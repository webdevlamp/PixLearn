import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist, cifar10
import os

def load_mnist_data(flatten=False):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    if not flatten:
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    
    print(f"MNIST Train: {x_train.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def load_cifar10_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"CIFAR-10 Train: {x_train.shape}, Test: {x_test.shape}")
    print(f"Classes: {cifar10_classes}")
    return (x_train, y_train), (x_test, y_test), cifar10_classes

def save_sample_images(x_test, y_test, dataset_name, num_samples=5, save_dir='data/samples'):
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(x_test[i].squeeze(), cmap='gray' if len(x_test[i].shape) < 3 else None)
        axes[i].set_title(f"Label: {y_test[i][0] if len(y_test[i].shape) > 0 else y_test[i]}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/{dataset_name}_samples.png')
    plt.close()
    print(f"Sample images saved to {save_dir}/{dataset_name}_samples.png")

if __name__ == "__main__":
    print("Loading MNIST...")
    (x_train, y_train), (x_test, y_test) = load_mnist_data()
    save_sample_images(x_test, y_test, 'mnist')
    
    print("\nLoading CIFAR-10...")
    (x_train, y_train), (x_test, y_test), classes = load_cifar10_data()
    save_sample_images(x_test, y_test, 'cifar10')
