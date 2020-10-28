# Kaggle Birdcall Identification

18th place solution for the [RSNA STR Pulmonary Embolism Detection](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection)

## Context

> In this competition, you’ll detect and classify pulmonary embolism (PE) cases. In particular, you'll use chest CTPA images (grouped together as studies) and your data science skills to enable more accurate identification of PE. If successful, you'll help reduce human delays and errors in detection and treatment.

For each study, an overal diagnosis consisting of 9 exam labels as well as whether or not the PE is present on each CT scan slice (image labels).
The metric is a weighted log loss of the several targets, explained [here](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/overview/evaluation). 

## Solution Overview

### Introduction

My solution consists of a two steps pipeline :  
- Image level efficientnet-b3, trained to classify whether the image has PE. This model is then used to extract features for each slice of the CT scan.
- A sequential is trained on the features extracted by the CNN, it predicts both the image and exam level labels, and directly optimizes the competition metric.

I joined the competition 9 days before the end, with the overall motivation of doing something similar to the winners of the previous RSNA competition. 
I was able to quickly build the CNN pipeline, and started training a bunch some models. The issue was that those models would be really long to train on my hardware (1x RTX 21080Ti), so I had to improvise.

Then, it was about quickly engineering the second part of the pipeline and the inference code, which was far from easy. 

Shortly after I joined, the overly powerful baseline was released. 
I did not end up really using any of the components of it, but it was an additional motivation for me to keep pushing.
I was able to come up with my first submission one day before the deadline, which *somehow* scored 23rd on the public leaderboard. 

### Data

As I could not fit the 900 Gb dataset on my computer, I solely relied on the [256x256 jpgs](https://www.kaggle.com/vaillant/rsna-str-pe-detection-jpeg-256) extracted by Ian Pan. 
Thanks a lot for making it possible for people like me to join the competition.

All the information to generate the data is available [here](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/discussion/182930). The windowing is the following :
- RED channel / LUNG window / level=-600, width=1500
- GREEN channel / PE window / level=100, width=700
- BLUE channel / MEDIASTINAL window / level=40, width=400

### First level : Convolutional Neural Networks

#### Undersampling

The issue with 2D images constructed from CT scans is that images are very similar. Therefore it makes sense not to use every slice per patient. 
As there is about 400 slices per patient on average, one epoch would take ages and this is not a path I wanted to go. 
Therefore, I only used 30 images per patient at each epoch, this is done using a custom sampler. 

Once this is done, I was able to train the models for 20 epochs in approximately 9 hours, using a 5-fold grouped by patient validation.

#### Models

Models were trained as part of a classical binary classification problem, using the binary cross-entropy
First experiments were conducted with a ResNeXt-50 model as it is usually a reliable baseline. 
I then tried to switch to a bigger ResNext-101, but results were not bigger so I quickly gave up with big architectures.
The last model I trained is an efficientnet-b3, which was chosen because a batch size of 32 could fit on my GPU. 
It performed slightly better so I sticked with this model, and I had no time left to train other models.

The efficientnet was trained for 15 epochs using a linear schedule with 0.05 warmup proportion. 

#### Augmentations

```
- albu.HorizontalFlip(p=0.5)
- albu.VerticalFlip(p=0.5)
- albu.ShiftScaleRotate(shift_limit=0.1, rotate_limit=45, p=0.5)
- albu.OneOf([albu.RandomGamma(always_apply=True), albu.RandomBrightnessContrast(always_apply=True),], p=0.5)
- albu.ElasticTransform(alpha=1, sigma=5, alpha_affine=10, border_mode=cv2.BORDER_CONSTANT, p=0.5)
```

### Second Level

#### Model

The model I used is a MLP + BidiLSTM one that predicts both the image and exam targets using the CNN extracted features as input. 
Two 2-layer classifiers are plugged on the concatenation of the output of the MLP and of the LSTM.
I used the concatenation of average and max pooling for the exam level targets.
In addition, multi-sample dropout was used for improved convergence


#### Training

The model was trained using the loss function that matches the metric. 
I also used stochastic weighted averaging for the last few epochs, once again to have a bit more robustness.
A single epoch took approximately a minute. 

The validation scheme is a normal 5-fold, and my CV scores were quite close to the 0.179 score I had on the public LB.


### Inference

My inference code is available [here](https://www.kaggle.com/theoviel/pe-inference-2). 
I used clipping to make sure the label assignment rules were respected, which dropped my score of approximately 0.003.


## Data

- Competition data is available on the [competition page](https://www.kaggle.com/c/rsna-str-pulmonary-embolism-detection/data)

You do not need to download the 900Gb of dicom files, only the `csv` files are used.

- The training images are available here : [256x256 jpgs](https://www.kaggle.com/vaillant/rsna-str-pe-detection-jpeg-256)

The relative location of the images `IMG_PATH` is specified in `params.py`, which can be adapted to your environment.


## Repository structure

- `input` : Input metadata expects to be here.
- `notebooks` : Notebook I used to experiment quickly on 2nd level models.
- `output` : Some outputs of the code.
- `src` : Source code.
- `logs` : Folder to save the logs to. Has to be created.

## Training

To retrain your own models, run the following scripts, which are in the `src` folder:

- `main.py` : Trains the first level models,
- `extract_features.py` : Extracts features using the trained models,
- `main_lvl2.py` : Trains second level models using extracted features.

Make sure to adapt the directories accordingly, to specify where to save and/or load the weights, features and images. 
They are either specified in the three scripts above or in `params.py`.

Batch sizes were chosen in order to fit in a single RTX 2080 Ti : 
- Training a CNN fold takes approximately 9 hours,
- Extracting the features takes 2 hours,
- Training a second level model fold takes about 15 minutes.

## Inference

- To reproduce our final score, fork this notebook [notebook](https://www.kaggle.com/theoviel/pe-inference-2) in the kaggle kernels.
- Model weights are available [on Kaggle](https://www.kaggle.com/theoviel/peweights/). Only the `efficientnet-b3` and `rnn_2` ones are used. 
