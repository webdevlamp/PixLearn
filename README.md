# PixLearn: Handwritten Digit Recognizer

[![GitHub stars](https://img.shields.io/github/stars/webdevlamp/PixLearn?style=social)](https://github.com/webdevlamp/PixLearn)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)

A modern, interactive **Handwritten Digit Recognizer** built with **Python 3.14**, **Scikit-Learn**, and **Gradio**. This project demonstrates a complete Machine Learning pipeline—from data processing to a user-friendly Web UI—without requiring heavy Deep Learning frameworks like TensorFlow.

## Features

-   **Interactive Web UI**: Built with Gradio, allowing users to upload images or draw digits directly in the browser.
-   **High Accuracy**: Uses a Support Vector Machine (SVM) with an RBF kernel, achieving ~98% accuracy on the MNIST dataset.
-   **Smart Preprocessing**: Automatically handles common user errors:
    -   **Color Inversion**: Converts black-on-white drawings to white-on-black (matching training data).
    -   **Auto-Centering**: Centers the digit in the frame to improve recognition regardless of where it's drawn.
-   **Python 3.14 Ready**: Fully compatible with the latest Python version, avoiding legacy dependency issues.

## Technologies Used

-   **Python 3.14**: Core language.
-   **Scikit-Learn**: Machine Learning library (SVM Classifier).
-   **Gradio**: Web interface for the model.
-   **NumPy & Pillow**: Image processing and array manipulation.
-   **OpenML**: Dataset fetching.

## How It Works

When you provide an input image (upload or draw), the application follows this flow:

1.  **Input**: The user provides an image via the Gradio Web UI.
2.  **Preprocessing (`app.py`)**:
    -   **Grayscale**: Converts the image to a single channel.
    -   **Resize**: Scales the image to 28x28 pixels (standard MNIST size).
    -   **Inversion**: Checks if the image is black-on-white. If so, it inverts colors to white-on-black.
    -   **Centering**: Detects the bounding box of the drawn digit and centers it within the 28x28 grid. This ensures the model sees the digit in the same position it was trained on.
    -   **Flattening**: Converts the 2D image array into a 1D vector (784 pixels) for the SVM.
3.  **Prediction (`src/models.py`)**:
    -   The flattened vector is fed into the trained **SVM (Support Vector Machine)** model.
    -   The model calculates the probability for each digit (0-9).
4.  **Output**: The app displays the predicted digit and the confidence score.

## Project Structure

```
PixLearn/
├── app.py                  # Gradio Web UI and inference logic
├── src/
│   ├── train.py            # Script to download data and train the SVM model
│   └── models.py           # Model definition (SVM Classifier)
├── data/
│   ── load_datasets.py    # Utilities for loading MNIST/CIFAR-10
├── models/                 # Directory for saved trained models
├── notebooks/              # Jupyter notebooks for exploration
├── python_basics/          # Python learning exercises
├── requirements.txt        # Project dependencies
└── README.md               # This file
```

## Getting Started

### Prerequisites

-   **OS**: Ubuntu 24.04 LTS (or similar Linux distribution).
-   **Python**: Version 3.14 (Required for this version of the project).

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/webdevlamp/PixLearn.git
    cd PixLearn
    ```

2.  **Install dependencies**:
    Since Python 3.14 is very new, you may need to use the `--break-system-packages` flag or install to the user directory.
    
    ```bash
    # Option A: User installation (Recommended)
    pip3 install --user -r requirements.txt
    
    # Option B: System-wide (If Option A fails)
    pip3 install --break-system-packages -r requirements.txt
    ```

### Usage

1.  **Train the Model**:
    Before running the app, you must train the SVM model on the MNIST dataset. This downloads the data and saves the model to `models/mnist_svm.pkl`.
    
    ```bash
    python3 src/train.py
    ```
    *Note: Training takes approximately 2-5 minutes.*

2.  **Run the Web App**:
    Start the Gradio server.
    
    ```bash
    python3 app.py
    ```

3.  **Open in Browser**:
    The terminal will display a local URL (e.g., `http://127.0.0.1:7860`). Open this link in your browser to use the Digit Recognizer.

## Command Line Prediction

You can also predict digits from the command line without the Web UI:

```bash
python3 src/predict.py /path/to/image.png
```

## Learning Journey

This repository documents a learning journey in Machine Learning, evolving from basic Python scripts to a functional AI application.

-   **Phase 1**: Python Basics (OOP, File I/O, Data Structures).
-   **Phase 2**: Data Exploration (MNIST, CIFAR-10).
-   **Phase 3**: Model Implementation (Random Forest -> SVM).
-   **Phase 4**: Web Integration (Gradio UI).

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
