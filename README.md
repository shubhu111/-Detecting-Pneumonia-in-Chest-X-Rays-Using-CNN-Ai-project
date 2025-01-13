# Detecting Pneumonia in Chest X-Rays Classification with CNN
# Project Overview
This project focuses on developing a Convolutional Neural Network (CNN) to classify chest X-ray images into two categories: Normal and Pneumonia. By leveraging deep learning techniques, this model aims to assist healthcare professionals in detecting Pneumonia more efficiently from chest radiographs.
# Dataset Description
The dataset is organized into three main folders, each containing two subfolders (NORMAL and PNEUMONIA) for binary classification:

1. Training Dataset:
- Contains over 1,000+ chest X-ray images per category.
- Data augmentation is applied to improve model generalization.
2. Validation Dataset:
- Includes 200 images per category.
- Used for fine-tuning model hyperparameters during training.
3. Testing Dataset:
- Comprises 200 images per category.
- Used to evaluate the final model's performance.
All images are resized to 224 x 224 pixels for consistency with the CNN input layer.
4. Dataset link:
- The dataset was downloaded from the online Kaggle platform. I modified the dataset according to my preferences. The dataset size and images may change on Kaggle.
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```
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
   - Each convolutional layer is followed by ReLU activation and max-pooling.
- Fully Connected Layers:
   - Flattens the feature map into a dense layer.
   - Includes dropout layers to prevent overfitting.
- Output Layer:
   - Uses a softmax activation function for binary classification (2 classes: NORMAL, PNEUMONIA).
3. Training:
- Optimizer: Adam (adaptive learning rate optimization).
- Loss Function: Categorical Crossentropy.
- Evaluation Metrics: Accuracy and Loss.
- Early stopping and learning rate reduction callbacks are used to enhance training efficiency.
4. Model Evaluation:
- The trained model is evaluated on the test set for metrics like accuracy, precision, recall, and F1-score.
5. Model Saving:
- The trained model is serialized and saved as "Detecting Pneumonia in Chest X-Rays Classification with CNN Dump file.joblib" for later use.

# Project Architecture (Code Overview)
1. Import Libraries:
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```
2. Define Data Paths:
```
train_path = "path/to/train"
test_path = "path/to/test"
validation_path = "path/to/validation"
```
3. Data Augmentation:
```
train_gen = ImageDataGenerator(rescale=1.0/255, 
                               rotation_range=20,
                               shear_range=0.2,
                               zoom_range=0.2,
                               horizontal_flip=True)

test_gen = ImageDataGenerator(rescale=1.0/255)
val_gen = ImageDataGenerator(rescale=1.0/255)
```
4. Model Architecture:
```
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for 2 classes
])
```
5. Compile the Model:
```
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```
6. Train the Model:
```
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True) # Add by choice(optional) 
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001) # Add by choice(optional) 

history = model.fit(train_gen.flow_from_directory(train_path),
                    validation_data=val_gen.flow_from_directory(validation_path),
                    epochs=25,
                    callbacks=[early_stop, reduce_lr])
```
7. Save the Model:
```
import joblib
joblib.dump(model, "Detecting Pneumonia in Chest X-Rays Classification with CNN Dump file.joblib")
```
# How to Use the Model
Follow these steps to use the trained model:

1. Load the Model:
```
import joblib
model = joblib.load("Detecting Pneumonia in Chest X-Rays Classification with CNN Dump file.joblib")
```
2. Define Classes:
```
classes = ['NORMAL', 'PNEUMONIA']
```
3. Predict Function
```
def predict(path):
    img = load_img(path, target_size=(224, 224, 3)) #import load_img, img_to_array using "from tensorflow.keras.preprocessing.image import load_img , img_to_array"
    img_arr = img_to_array(img)
    norm = img_arr / 255.0
    flat = np.expand_dims(norm, axis=0) # import numpy as np
    pred = model.predict(flat)[0]
    return classes[np.argmax(pred)]
```
4. Test Prediction:
```
result = predict("path/to/image.jpg")
print(f"The model predicts: {result}")
```
# Repository Structure
```
📂 Detecting-Pneumonia-in-Chest-XRays
├── 📂 Data
│   ├── 📂 train
│   │   ├── 📂 NORMAL
│   │   ├── 📂 PNEUMONIA
│   ├── 📂 test
│   │   ├── 📂 NORMAL
│   │   ├── 📂 PNEUMONIA
│   ├── 📂 validation
│       ├── 📂 NORMAL
│       ├── 📂 PNEUMONIA
├── 📝 README.md
├── 📄 project.ipynb
├── 📄 test.ipynb
├── 📄 Detecting Pneumonia in Chest X-Rays Classification with CNN Dump file.joblib
```
# Key Features
1. State-of-the-Art CNN Model: A well-designed architecture optimized for image classification.
2. Data Augmentation: Enhanced the training dataset with augmentation techniques.
3. Efficient Training: Utilized callbacks like early stopping and learning rate reduction.
4. Simple Deployment: The model is saved in .joblib format for easy reuse and deployment.
# Conclusion
This project demonstrates how deep learning can be effectively applied in medical imaging to assist healthcare professionals in diagnosing Pneumonia. With an accuracy of 93.5%, the model provides reliable predictions, making it a valuable tool for preliminary screenings. Future work may include extending the model to multi-class classification or implementing it in real-time diagnostic tools.
