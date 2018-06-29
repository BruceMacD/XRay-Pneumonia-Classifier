# XRay-Pneumonia-Classifier

## Project Description
The goal of this project is so classify the presence of pneumonia in a patient based off their chest x-ray. There are 3 different examples in the dataset (normal, bacterial pneumonia, and viral pneumonia). Our initial goal is to classify the yes/no question for if a patient has malaria. We will research a variety of models to see what is the most effective.

## Dataset
Available on Kaggle [1]. 5863 labelled x-ray images from a frontal view. There are 2 categories of normal/pneumonia.

## Milestones
* July 9, 2018
1. Create a basic CNN classifier that has an acceptable (about 70%) accuracy on test data.
* July 19, 2018
2. Try different types of convolutions and note the effect on test accuracy.
    * Convolution layer parameters -> kernel size, stride, padding.
    * Dilated convolutions
    * Transposed convolutions
    * Separable convolutions
* July 30, 2018
3. Try implementing an experimental AllCNN, a CNN consisting of solely convolutional layers. 
 

Notes:
* Look into image processing (already greyscale but could use sharpening or something)
* Do a basic CNN implementation, try some different hidden layer configurations
* Look into AllCNN - https://arxiv.org/abs/1412.6806#
* An Introduction to different Types of Convolutions in Deep Learning - https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

## Work Log
Keep track of hours here:

## Sources (TODO: formatting)
1 - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
