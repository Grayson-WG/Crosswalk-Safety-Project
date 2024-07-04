# Crosswalk-Safety-Project
This code is used to determine whether or not it is safe to cross the street.

## The Algorithm
This algorithm uses the Semantic Segmentation with SegNet, and uses the pretrained cityscapes model to detect cars and other vehicles to determine whether it is safe to cross the street on a live video feed.

## Running This Project
### Jetson-Inference Library
To set up this project you first need to have the Jetson-Inference Library

#### Step 1
You first need to connect to your nano using SSH, and then open a new terminal

#### Step 2
You need to then install cmake and git using these commands: sudo apt-get update sudo apt-get install git cmake

#### Step 3
Next, you need to clone Jetson-Inference with this command: git clone --recursive https://github.com/dusty-nv/jetson-inference

### Python Packages
Now that you have the Jetson-Inference library, you now need to install numpy, opencv and argparse.

#### Step 1
Open a new terminal

#### Step 2
To install numpy and argparse you will need these commands: pip install --upgrade pip pip install numpy pip install argparse pip install opencv-python

#### Step 3
If you want to check to see if they are installed run these commands: python -c "import numpy; print(numpy.version)" python -c "import argparse; print(argparse.file)" python -c "import cv2; print(cv2.version)"

### Option 2
If you don't want to use pip you can also use apt(Advanced Package Tool)

#### Step 1
To install them on apt you need to use these commands: sudo apt install python3-numpy sudo apt install python3-argparse sudo apt install python3-opencv

#### Step 2
If you want to check to see if they are installed run these commands: python -c "import numpy; print(numpy.version)" python -c "import argparse; print(argparse.file)" python -c "import cv2; print(cv2.version)"
