# XRay-Pneumonia-Classifier

## Project Description
The goal of this project is so classify the presence of pneumonia in a patient based off their chest x-ray. There are 3 different examples in the dataset (normal, bacterial pneumonia, and viral pneumonia). Our initial goal is to classify the yes/no question for if a patient has pneumonia. We will research a variety of models to see what is the most effective.

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
* [Running the Kaggle kernal is the easiest method of runnning this notebook.](https://www.kaggle.com/brucemacd/detecting-pneumonia-in-x-ray-images)
* If you are running the notebook locally you must download the [data source](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) and extract it to the root of the directory of this repo.
 

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

Notes from second meeting with Dr. Oore:
* Project due week after classes are done (August 7)
* Try stripping layers from pre-trained model, with a fully connected layer at the end
* Take of different amounts of layers, freeze the pretrained layers and train the new ones
* Data augmentation, we can do rotation, augmentation gives more examples.
* Note why shaprening augmentation didnt work 
* Yes/No feature pre-processing
* Different nets may have different cielings
* Ensemble nets (vgg16 and simple net)


## Resources (TODO: formatting)
### General Resources
1 - https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

2 - https://cs231n.github.io/neural-networks-1/

3 - https://github.com/bendidi/X-ray-classification

4 - https://lhncbc.nlm.nih.gov/system/files/pub9175.pdf

5 - https://arxiv.org/abs/1803.02315

6 - http://www.steves-digicams.com/knowledge-center/brightness-contrast-saturation-and-sharpness.html (for data augmentation)

7 - https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html

8 - https://arxiv.org/pdf/1709.09902.pdf

9 - https://arxiv.org/pdf/1502.03167.pdf%E7%9A%84paper%E9%80%82%E5%90%88%E6%83%B3%E6%B7%B1%E5%85%A5%E4%BA%86%E8%A7%A3%E5%8E%9F%E7%90%86%EF%BC%8C%E8%BF%99%E4%B8%AA%E8%A7%86%E9%A2%91%E5%BE%88%E6%B8%85%E6%A5%9A%E7%9A%84%E8%AE%B2%E4%BA%86bn%E8%B5%B7%E5%88%B0%E7%9A%84%E4%BD%9C%E7%94%A8%E3%80%82

10 - Seperable Conv - https://arxiv.org/abs/1610.02357

11 - where we got a bunch of code: https://mc.ai/detect-pneumonia-from-x-ray-using-convolutional-neural-network/

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
