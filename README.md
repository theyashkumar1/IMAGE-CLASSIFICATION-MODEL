# IMAGE-CLASSIFICATION-MODEL

COMPANY: CodTech IT Solutions

NAME: Yash Kumar

INTERN ID: CT06DG602

DOMAIN: Machine Learning

DURATION: 6 Weeks 

MENTOR: Neela Santosh

🐶🐱 Dogs vs Cats Classifier using CNN
This project presents a deep learning-based solution to a classic computer vision challenge: classifying images of dogs and cats. Built using TensorFlow and Keras, it uses a Convolutional Neural Network (CNN) to accurately distinguish between dog and cat images with significant generalization and performance. This end-to-end project covers data loading, preprocessing, model building, training, evaluation, visualization, and prediction.

📂 Dataset
The dataset used in this project is the Dogs vs Cats dataset from Kaggle. It contains 25,000 labeled images (12,500 dogs and 12,500 cats) for training and 5,000 for testing.

Steps:

Dataset is downloaded directly using the Kaggle API.

Data is organized into train and test directories automatically inferred by Keras’ image_dataset_from_directory.

🧠 Model Architecture
The model is a deep CNN implemented using the Sequential API. It includes:

Multiple Conv2D layers for feature extraction

Batch Normalization to stabilize and accelerate training

MaxPooling to reduce spatial dimensions

Dropout layers to combat overfitting

A final Dense layer with sigmoid activation for binary classification

Model Summary:

Input size: (256, 256, 3)

Parameters: ~14.8 million

Activation Functions: ReLU for hidden layers, Sigmoid for output

⚙️ Training & Optimization
Loss Function: binary_crossentropy

Optimizer: adam

Metrics: accuracy

Epochs: 10

Image Preprocessing: Normalization using .map() function

Throughout training, the model shows improvement in both training and validation accuracy, reaching up to 78%+ validation accuracy after just 10 epochs.

📉 Results & Visualization
The training history is visualized using Matplotlib:

Accuracy and loss curves for both training and validation sets

Helps identify overfitting and underfitting trends

📸 Custom Prediction
After training:

Real-world dog and cat images were used to test the model

Images are preprocessed (resized and reshaped)

The model predicts labels (0 for cat, 1 for dog)

Final predictions are correct
