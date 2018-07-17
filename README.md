# [An intriguing failing of convolutional neural networks and the CoordConv solution](https://arxiv.org/pdf/1807.03247.pdf)

Implementation of the CoordConv (Convolution and Deconvolution) for a Variational Autoencoder applied to the MNIST dataset

## 1. Install dependencies:
```
pip install -r requirements.txt
```

## 2. Run training without coordConv layers
```
python trainVAE.py --useCoordConv False
```

## 3. Run training with coordConv layers
```
python trainVAE.py --useCoordConv True
```

## 4. Display Reconstructions on TensorBoard
```
tensorboard --logdir logdir
```

## 5. Compare reconstructions:
```
python compare_generated_images.py
```
