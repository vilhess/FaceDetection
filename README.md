# Computer Vision Project - Face Position Detection

This project utilizes the PyTorch Faster R-CNN model to detect the position of people's heads in images.

## Objective

The main goal of this project is to achieve accurate head position detection by fine-tuning the Faster R-CNN model on a specific dataset.

## Installation

1. Clone this repository to your local machine:

```bash
git clone https://github.com/vilhess/FaceDetection.git
cd FaceDetection
```
2. Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset is a dataset composed of face images associated with their bounding boxes.

## Usage

A training script is provided to train the model on a custom dataset. The script can be run as follows:

```bash
python training.py
```
The inference script can be run as follows:

```bash
python testing.py
```


