# Project: Recognize Emotions from Facial Images with Convolutional Neural Networks

## Overview

This project contains Allison Dzubak's models for recognizing emotions from facial images with convolutional neural networks. The dataset was provided by the MIT Applied Data Science Program.

## Data

The data set is available in a compressed format in '/data/Facial_emotion_images.zip' at 34 MB. The validation and test images were predetermined by the MIT Program.

- Notebook '0-Data-Exploration.ipynb' unpacks the zip file to '/data/Facial_emotion_images' directory with size 83 MB. 
- Notebook '1-Data-Preprocessing-Facial-Detection.ipynb' creates '/data/face_detection' directory to store preprocessed data following facial detection screening.
- Notebook '2-Data-Preprocessing-Remove-Duplicates.ipynb' creates '/data/unique_images' directory to store preprocessed data after removing duplicate images.

After data preprocessing, the data directory contains:
- Facial_emotion_images.zip 	# Original dataset
- Facial_emotion_images/	# Original dataset unpacked
- face_detection/		# Dataset after facial detection screening
- unique_images/		# Dataset after removing duplicates

Each of the dataset directories has 'test', 'train', and 'validation' subdirectories. 
Each of the 'test', 'train', and 'validation' has 4 subdirectories: 'happy', 'neutral', 'sad', 'surprise'.   

## Models

Three pre-trained model architectures were used as a starting point: VGG16, ResNet50V2, and EfficientNetB0. 

For each of the pre-trained model architectures:

- (Model 0) Test the pre-trained model essentially 'off-the-shelf'. Take the pre-trained model, freeze the convolutional bottom, then add a trainable top that has the same structure as the original model. Use the default Adam learning rate of 0.001.
- (Model 1) Test the base model (Model 0) and include a low level of data augmentation (illustrated in 5a). Use the default Adam learning rate of 0.001.
- (Model 2) Test the base model (Model 0), including the low level of data augmentation, and unfereze the final convolution block of the pre-trained model (keeping all the earlier blocks still frozen) while decreasing the Adam optimizer learning rate to 0.0001.
- (Model 3) Test the base model (Model 0), increase the level of data augmentation to a high level of augmentation, unfreeze the final convolution block and use the smaller Adam learning rate of 0.0001.
- (Model 4) Test the base model (Model 0), using the high level of data augmentation, and unfreeze the top 2 convolution blocks with the learning rate of 0.0001.
- (Model 5) Test the bast model (Model 0), using the high level of data augmentation, unfreeze the top 2 convolution blocks and add a dense layer with the learning rate of 0.0001.
- (Model 6) Test the base model (Model 0), using the high level of data augmentation, unfreeze the top 2 convolution blocks, add a dense layer, and decrease the learning rate further to 0.00001.
- (Model 7) Test the base model (Model 0), and decrease the level of data augmentation to somewhere between low and high - a mid level of augmentation. Unfreeze the top 2 convolution blocks, add a dense layer, and use a learning rate of 0.00001.
- (Model 8) Test the base model (Model 0), using the mid level data augmentation, unfreeze the top 3 convolution blocks, add a dense layer, and use a learning rate of 0.00001.
- (Model 9) Test the base model (Model 0), using the mid level data augmentation, unfreeze all convolution blocks, and use a learning rate of 0.00001.
- (Model 10) Test the base model (Model 0), using the mid level data augmentation, unfreeze all convolution blocks, and increase the batch size from 32 to 128 with a learning rate of 0.00001.
- (Model 11) Test the base model (Model 0), using the mid level data augmentation, unfreeze all convolution blocks, increase the batch size to 128, and balance the class weights, and use a learning rate of 0.00001.

A visual summary of these model variations is shown in '/images/model_versions.png'

Model checkpoint files are stored temporarily in the Google Colab temporary filesystem but not stored permanently. If you wish to store model weights, transfer model weight files from temporary filesystem to main_directory/models in '4-Model-Training.ipynb' notebooks.

Notebook '5-Final-Model-Training-Storage.ipynb' stores model weights for the final selected model of each pre-trained architecture in '/models' (file size > 100MB) 

## Notebooks

All notebooks contain code in the first cell for running in Google Colab. 
The main working directory is set in the Google Colab environment as '/content/drive/MyDrive/facial-emotion-detection-cnn/'. 
All subsequent working and storage directories/files/paths are specified relative to the main directory. 
Model training notebooks (4a, 4b, 4c, 5) set the location for the temporary Colab filesystem for faster access to data during training.
To run locally, comment out the Colab-specific cells and set your preferred 'main_directory'.


### 0-Data-Exploration.ipynb 
This notebook contains exploratory data analysis. 
Performed here:

- Explore dataset size and distribution of images by emotion class
- Visualize a random sampling of dataset images

### 1-Data-Preprocessing-Facial-Detection.ipynb
This notebook contains the first step of data preprocessing. Performed here:

- Use a facial detection model to detect if a face is present in the image
- Store images where a face was detected for subsequent preprocessing data cleaning 

### 2-Data-Preprocessing-Remove-Duplicates.ipynb
This notebook contains the second step of data preprocessing. Performed here:

- Use image hashing to determine if any duplicate images are present in the dataset
- Store unique (duplicates dropped) images for subsequent data processing 

### 3-View-Data-Augmentations.ipynb
This notebook contains visualizations of the data augmentations used.

### 4-Model-Training.ipynb
- 4a-VGG16-Model-Training.ipynb contains the model training using the VGG16 architecture
- 4b-ResNet50V2-Model-Training.ipynb contains the model training using the ResNet50V2 architecture
- 4c-EfficientNetB0-Model-Training.ipynb contains the model training using the EfficientNetB0 architecture

Performed in each notebook:

- Setup content structure in temporary Colab filesystem
- Transfer preprocessed data to temporary Colab filesystem for faster access during training
- Train each model according to specifications described in 'Models'
- Plot training and validation accuracies per epoch
- Show confusion matrix and classification report
- Visualize misclassified images

### 5-Final-Model-Training-Storage.ipynb
This notebook takes the most promising model from 4a, 4b, and 4c and allows more training epochs. Model weights (best only) are stored in '/models'.

### 6-Testing-Unseen-Data.ipynb
This notebook contains the final model testing on the unseen test data. Performed here:

- Load VGG16 Model 7 weights and specify architecture
- Make predictions on test data
- Show confusion matrix and classification report
- Visualize misclassified images

## Images
- model_versions.png shows the model training variations
- model_accuracies.png shows the training/validation performace of all models
- emotion_class_performance.png shows the emotion class performance of the best performing models

