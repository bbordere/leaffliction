# Leaffliction
Computer vision project for plant leaf diseases.
## Overview
This project implements a Convolutional Neural Networks (CNN) model to detect and classify leaf diseases from images. The pipeline includes data processing, class balancing, feature extraction and training of a classification model.

## Data Analysis
The first step is to analyze the dataset, particularly to determine the distribution of the classes present. This is accomplished using the **Distribution.py** program.

![Distrib](https://github.com/user-attachments/assets/6913d811-f58d-423b-9b8e-230e3f72ab8c)

## Data Augmentation
As seen above, the dataset is clearly unbalanced. To address this, we apply data augmentation techniques such as flipping, rotating, blurring, adjusting contrast, and more. This is handled by the **Augmentation.py** program.

![Augment](https://github.com/user-attachments/assets/e568eaf2-7b2d-47cb-8507-051e6b15ebf4)

## Data Transformation
After balancing the dataset, the images must be transformed to facilitate feature extraction. This is the role of the **Transformation.py** program. It leverages the [PlantCV](https://plantcv.readthedocs.io/en/stable/) and [OpenCV](https://opencv.org/get-started/) libraries. </br>
Filters such as **Gaussian Blur**, **Region of Interest (ROI)**, and **Canny Edge Detection** are applied directly to the images to create the final dataset.
![Transf](https://github.com/user-attachments/assets/d0fb3f8f-9871-41ee-86b3-0984a8434a48)

## Classification
Once the final dataset has been generated, we need to use it to train our model. It will use the augmented and filtered images to learn the characteristics of specified leaf diseases using **Convolutional Neural Networks** with [Tensorflow](https://www.tensorflow.org/) library. This is achieved by the **train.py** program.
</br>
</br>
After having trained our model, we can use it to make predictions
The **predict.py** program take a leaf image as input, applies transformations, and identifies the type of disease present
