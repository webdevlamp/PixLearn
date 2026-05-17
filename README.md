# PixLearn: Image Classification using Deep Learning

[![GitHub stars](https://img.shields.io/github/stars/webdevlamp/PixLearn?style=social)](https://github.com/webdevlamp/PixLearn)
[![GitHub forks](https://img.shields.io/github/forks/webdevlamp/PixLearn?style=social)](https://github.com/webdevlamp/PixLearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A deep learning project focused on image classification using Python, TensorFlow, and Keras. This repository documents my learning journey as I explore the world of computer vision and AI.

## Project Goals

- Implement image classification models using deep learning techniques
- Experiment with popular architectures and techniques
- Improve accuracy and efficiency using data augmentation and transfer learning
- Learn and document the entire ML pipeline from data loading to model deployment

## Technologies Used

- Python 3.12
- TensorFlow 2.x
- Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Getting Started

### Prerequisites

- Ubuntu 24.04 LTS (or similar Linux distribution)
- Python 3.12

### Installation

```bash
# Clone the repository
git clone https://github.com/webdevlamp/PixLearn.git
cd PixLearn

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

See [install.md](install.md) for detailed installation instructions.

## Project Structure

```
PixLearn/
├── data/                       # Dataset loading and preprocessing
│   ├── __init__.py
│   └── load_datasets.py        # MNIST & CIFAR-10 loaders
├── src/                        # Source code
│   ├── __init__.py
│   ├── models.py               # CNN & transfer learning models
│   ├── train.py                # Training scripts
│   ├── evaluate.py             # Model evaluation & metrics
│   └── data_augmentation.py    # Data augmentation utilities
├── notebooks/                  # Jupyter notebooks
│   ├── 01_tensorflow_keras_basics.ipynb
│   ├── 02_opencv_basics.ipynb
│   └── 03_visualization_results.ipynb
├── models/                     # Saved models and outputs
├── python_basics/              # Python learning exercises
│   ├── test.py
│   ├── calculator.py
│   ├── guessinggame.py
│   ├── todos.py
│   ├── file_io.py
│   ├── oop_demo.py
│   ├── functions_modules.py
│   └── data_structures_algorithms.py
├── requirements.txt
├── install.md
└── README.md
```

## Usage

### 1. Explore Learning Notebooks

Start with the Jupyter notebooks to understand the basics:

```bash
jupyter notebook notebooks/
```

- **01_tensorflow_keras_basics.ipynb** - TensorFlow tensors, building and training neural networks
- **02_opencv_basics.ipynb** - Image processing, filtering, contours, thresholding
- **03_visualization_results.ipynb** - Model evaluation, confusion matrices, feature maps

### 2. Run Python Basics Exercises

```bash
python python_basics/file_io.py
python python_basics/oop_demo.py
python python_basics/functions_modules.py
python python_basics/data_structures_algorithms.py
```

### 3. Load and Explore Datasets

```bash
python data/load_datasets.py
```

### 4. Train Models

```bash
# Train CNN on MNIST and CIFAR-10
python src/train.py

# Train with transfer learning (MobileNetV2, ResNet50, etc.)
python src/transfer_learning.py

# Run data augmentation examples
python src/data_augmentation.py
```

### 5. Evaluate Models

```bash
python src/evaluate.py
```

## Model Architectures

### CNN for MNIST
- 3 Conv2D layers (32, 64, 64 filters)
- MaxPooling after first two conv layers
- Dense layer (64 units) + Softmax output

### CNN for CIFAR-10
- 4 Conv2D layers with BatchNormalization
- Dropout for regularization
- Data augmentation (flip, rotation, zoom, translation)

### Transfer Learning
- MobileNetV2, ResNet50, EfficientNetB0 backbones
- Two-phase training: frozen base then fine-tuning
- GlobalAveragePooling2D + Dense classifier head

## Learning Roadmap (GitHub Issues)

This project follows a structured learning path tracked via GitHub Issues:

- [x] Python Basics (OOP, Functions, File I/O, DSA)
- [x] TensorFlow/Keras Basics
- [x] OpenCV Computer Vision Basics
- [x] Explore Datasets (MNIST, CIFAR-10)
- [x] Implement Image Classification Models
- [x] Train and Evaluate Models
- [x] Visualize Results
- [x] Experiment with Transfer Learning & Data Augmentation

## Results

Training results, confusion matrices, and prediction visualizations are saved to the `models/` directory after training:

- `models/mnist_training_history.png` - MNIST training curves
- `models/cifar10_training_history.png` - CIFAR-10 training curves
- `models/*_confusion_matrix.png` - Confusion matrices
- `models/*_predictions.png` - Prediction visualizations

## Contributions and Feedback

Contributions, feedback, and suggestions are welcome! If you have any ideas or want to collaborate, please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Follow My Journey

Follow my progress, insights, and mistakes as I learn and improve. Stay tuned for updates!
