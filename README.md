# Detecting Pneumonia in Chest X-Rays Classification with CNN
## Project Overview
This project focuses on developing a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories: Normal and Pneumonia. By leveraging deep learning techniques, this model aims to assist healthcare professionals in detecting Pneumonia more efficiently from chest radiographs.
## Dataset Description
The dataset is organized into three main folders, each containing two subfolders (NORMAL and PNEUMONIA) for binary classification:

1. Training Dataset:
- Contains over 1,000 chest X-ray images per category.
- Data augmentation is applied to improve model generalization.
2. Validation Dataset:
- Includes 200 images per category.
- Used for fine-tuning model hyperparameters during training.
3. Testing Dataset:
- Comprises 200 images per category.
- Used to evaluate the final model's performance.
All images are resized to 224 x 224 pixels for consistency with the CNN input layer.
## Project Workflow
The project follows a structured pipeline for model development:
1. Data Preprocessing:
- The ImageDataGenerator class is used to:
- Rescale pixel values to the range [0, 1].
- Apply augmentation techniques like rotation, shear, width/height shift, zoom, and horizontal flips.
2. Model Architecture:
- Input Layer: Resizes images to 224×224 with 3 channels (RGB).
- Convolutional Layers:
   -  Extract spatial features using 2D convolution layers.
Each convolutional layer is followed by ReLU activation and max-pooling.
Fully Connected Layers:
Flattens the feature map into a dense layer.
Includes dropout layers to prevent overfitting.
Output Layer:
Uses a softmax activation function for binary classification (2 classes: NORMAL, PNEUMONIA).
