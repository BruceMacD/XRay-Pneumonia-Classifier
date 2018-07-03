# XRay-Pneumonia-Classifier

## Project Description
The goal of this project is so classify the presence of pneumonia in a patient based off their chest x-ray. There are 3 different examples in the dataset (normal, bacterial pneumonia, and viral pneumonia). Our initial goal is to classify the yes/no question for if a patient has malaria. We will research a variety of models to see what is the most effective.

## Dataset
Available on Kaggle [1]. 5863 labelled x-ray images from a frontal view. There are 2 categories of normal/pneumonia.

## Milestones
* July 11, 2018
1. Find a pre-trained neural net.
* July 19, 2018
2. Strip off the last few layers of the model and specialize them.
* July 30, 2018
3. Find some related work on weird stuff. 
 

Notes:
* Look into image processing (already greyscale but could use sharpening or something)
* Do a basic CNN implementation, try some different hidden layer configurations
* Look into AllCNN - https://arxiv.org/abs/1412.6806#
* An Introduction to different Types of Convolutions in Deep Learning - https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

Notes from meeting with Dr. Oore:
Get a pre-trained net, strip off last few layers and specialize them.
Edge filters will be prelearned. 
By adding a tiny bit where did we get on competition spread.
Using a fraction off the data what is the impact/risk on model?
If you do weird things show you looked for realted work.
Fair bit of related work stuff.

## Work Log
Keep track of hours here:

## Sources (TODO: formatting)
1 - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
