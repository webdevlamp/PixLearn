import os
import numpy as np
import urllib.request
import pickle
import gzip
from sklearn.datasets import fetch_openml

def load_mnist_data(flatten=False):
    print("Downloading MNIST via OpenML (may take a moment)...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    x_all = mnist.data.astype('float32') / 255.0
    y_all = mnist.target.astype('int32')
    
    x_train, x_test = x_all[:60000], x_all[60000:]
    y_train, y_test = y_all[:60000], y_all[60000:]
    
    if not flatten:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
    
    print(f"MNIST Train: {x_train.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)

def load_cifar10_data():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    data_dir = os.path.join(os.path.dirname(__file__), 'cifar-10-batches-py')
    
    if not os.path.exists(data_dir):
        print("Downloading CIFAR-10...")
        import tarfile
        import io
        req = urllib.request.urlopen(url)
        with tarfile.open(fileobj=io.BytesIO(req.read()), mode='r:gz') as tar:
            tar.extractall(path=os.path.dirname(__file__))
    
    def load_batch(filename):
        filepath = os.path.join(data_dir, filename)
        with open(filepath, 'rb') as f:
            entry = pickle.load(f, encoding='latin1')
        return entry['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1), np.array(entry['labels'])
    
    x_train_parts, y_train_parts = [], []
    for i in range(1, 6):
        x, y = load_batch(f'data_batch_{i}')
        x_train_parts.append(x)
        y_train_parts.append(y)
    
    x_train = np.vstack(x_train_parts).astype('float32') / 255.0
    y_train = np.concatenate(y_train_parts).astype('int32')
    
    x_test, y_test = load_batch('test_batch')
    x_test = x_test.astype('float32') / 255.0
    y_test = y_test.astype('int32')
    
    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
    
    print(f"CIFAR-10 Train: {x_train.shape}, Test: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test), cifar10_classes

def save_sample_images(x_test, y_test, dataset_name, num_samples=5, save_dir='data/samples'):
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    for i in range(num_samples):
        axes[i].imshow(x_test[i].squeeze(), cmap='gray' if len(x_test[i].shape) < 3 else None)
        axes[i].set_title(f"Label: {y_test[i]}")
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
