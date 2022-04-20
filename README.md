# CSC413 Final Project: MultiColumn Convolutional Recurrent Neural Network

A novel architecture for genre classification that uses multi-column CRNNs.

## Report

The paper for this code can be found [here](https://www.overleaf.com/read/cfvfdmmbwqwb). 

## Dataset:

Dataset is retrieved from [GTZAN](http://marsyas.info/downloads/datasets.html)

To download and extract dataset:

`wget http://opihi.cs.uvic.ca/sound/genres.tar.gz`

`tar -xvzf genres.tar.gz`

## Training

To train models, run:

`python train.py [convnet, crnn, mccrnn]`

The model weights, loss and accuracy curves will be saved in a `results` folder.

## Evaluation

To evaluate models, run:

`python evaluate.py [convnet, crnn, mccrnn]`
