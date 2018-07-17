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

## Installation:
* Go look at the Kaggle notebook, this repo is just for reference.
 

Notes:
* Look into image processing (already greyscale but could use sharpening or something)
* Do a basic CNN implementation, try some different hidden layer configurations
* Look into AllCNN - https://arxiv.org/abs/1412.6806#
* An Introduction to different Types of Convolutions in Deep Learning - https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

Notes from meeting with Dr. Oore:
* Get a pre-trained net, strip off last few layers and specialize them.
* Edge filters will be prelearned. 
* By adding a tiny bit where did we get on competition spread.
* Using a fraction off the data what is the impact/risk on model?
* If you do weird things show you looked for related work.
* Fair bit of related work stuff.

## Work Log
Keep track of hours here:

* July 5 - 2 hours - Figuring out Kaggle kernals. 
* July 10 - 1.5 hours - Annotating notebook while gathering information on models and techniques currently implemented.
* July 11 - 2 hours - More refactoring and trying out a new top-layer on the model.
* July 12 - 2 hours - Reading on different output activations, trying some different training configurations.
* July 13 - 2 hours (Seth) - Reading about removing layers from an NN, researching guided backprop and transfer learning
* July 13 - 2.5 hours (Bruce) - Reading about transfer learning (which we are already doing), implementing and testing sharpening input images
* July 14 - 1.5 hours (Bruce) - Noticed that resnet50 was suggested as a popular solution for x-ray classification in papers[4]. Implemented a restnet50 classifier with transfer learning, but it was not very successful.
* July 16 - 1 hour (Seth) - Still trying to understand guided backpropagation. Found several resources.

## Resources (TODO: formatting)
### General Resources
1 - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

2 - https://cs231n.github.io/neural-networks-1/

3 - https://github.com/bendidi/X-ray-classification

4 - https://lhncbc.nlm.nih.gov/system/files/pub9175.pdf

5 - https://arxiv.org/abs/1803.02315

6 - http://www.steves-digicams.com/knowledge-center/brightness-contrast-saturation-and-sharpness.html (for data augmentation)

7 - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

### To explore
1 - https://jacobgil.github.io/deeplearning/pruning-deep-learning

2 - http://ruder.io/transfer-learning/  (This is basically what we're trying to do. Good to use from report; endorsement from Andrew Ng)

3 - https://arxiv.org/pdf/1712.07632.pdf preprocessing xrays by removing bones to increase accuracy

4 - https://stats.stackexchange.com/questions/191397/understanding-probabilistic-neural-networks Probablistic neural networks. 

#### Guided Backprop
1 - http://www.cs.toronto.edu/~guerzhoy/321/lec/W07/HowConvNetsSee.pdf (Slide 14, Guided Backprop). Could be useful in getting our network to "assume" certain information about the xrays, like the fact that they are likely to contain pneumonia victims

2 - https://arxiv.org/pdf/1412.6806.pdf The authority on gbp

3 - https://github.com/Lasagne/Recipes/blob/master/examples/Saliency%20Maps%20and%20Guided%20Backpropagation.ipynb GBP with lasagne

4 - https://ramprs.github.io/2017/01/21/Grad-CAM-Making-Off-the-Shelf-Deep-Models-Transparent-through-Visual-Explanations.html   GBP + GRAD-CAM explanation
