Installation Guide
Prerequisites- Ubuntu 24.04.3 LTS
- Python 3.12.3

Install Python and Required Libraries
bash
# Update package list
sudo apt update

# Install python3-pip
sudo apt install python3-pip

# Install python3.12-venv
sudo apt install python3.12-venv

# Create virtual environment
python3 -m venv myenv

# Activate virtual environment
source myenv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required libraries
pip install tensorflow keras opencv-python numpy pandas


Verify Installation
bash
# Verify TensorFlow installation
python -c "import tensorflow as tf; print(tf.__version__)"

# Verify Keras installation
python -c "import keras; print(keras.__version__)"

# Verify OpenCV installation
python -c "import cv2; print(cv2.__version__)"


Deactivate Virtual Environment
bash
# Deactivate virtual environment
deactivate
