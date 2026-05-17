# Installation Guide

This guide covers setting up the PixLearn project on a local machine.

## Prerequisites

-   **Operating System**: Ubuntu 24.04 LTS (or similar Linux distribution).
-   **Python**: Version **3.14** is required for this version of the project.
    -   Verify installation: `python3 --version`

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/webdevlamp/PixLearn.git
cd PixLearn
```

### 2. Install Dependencies

This project uses **Scikit-Learn** and **Gradio**. Since Python 3.14 is a bleeding-edge version, package managers might restrict system-wide installations. Use one of the following methods:

**Method A: User Installation (Recommended)**
Installs packages for your user account only.
```bash
pip3 install --user -r requirements.txt
```

**Method B: Break System Packages**
If Method A fails or packages are not found, use this flag to allow installation.
```bash
pip3 install --break-system-packages -r requirements.txt
```

### 3. Verify Installation

Ensure the key libraries are installed correctly:

```bash
# Check Scikit-Learn
python3 -c "import sklearn; print(f'Scikit-Learn: {sklearn.__version__}')"

# Check Gradio
python3 -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

## Running the Project

### 1. Train the Model
You must train the model before using the application. This script downloads the MNIST dataset and trains an SVM classifier.

```bash
python3 src/train.py
```
*Output:* You should see `Test Accuracy: ~0.98` and `Model saved to models/mnist_svm.pkl`.

### 2. Launch the Web App
Start the interactive interface:

```bash
python3 app.py
```
*Output:* The terminal will show a local URL (e.g., `http://127.0.0.1:7860`). Open this in your browser.

## Troubleshooting

-   **ModuleNotFoundError**: Ensure you are running commands from the project root directory (`/home/horizon/projects/PixLearn`).
-   **Python Version Error**: This project is optimized for Python 3.14. If you use an older version, some dependencies might conflict.
-   **Permission Errors**: If `pip` complains about externally managed environments, use the `--user` or `--break-system-packages` flags as shown above.
