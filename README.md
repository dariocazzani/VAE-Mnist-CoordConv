# [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/pdf/1807.03247.pdf)

Implementation of the CoordConv (Convolution and Deconvolution) for a Variational Autoencoder applied to the MNIST dataset

## Install dependencies:
```
pip install -r requirements.txt
```

## Run training for both teacher and student
```
python trainVAE.py
```

## Display Reconstructions on TensorBoard
```
tensorboard --logdir logdir
```
