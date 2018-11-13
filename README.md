# CVAE-Tensorflow (2018/11/12)
## Introduction
I implement a tensorflow model of a Conditional Variational Autoencoder
- I use mnist dataset as training dataset.

## Environment
- Ubuntu 16.04
- Python 3.5

## Depenency
- Numpy
- matplotlib

## Files
- cvae.py : Model definition.
- main.py : Execute training and pass the default value.
- train.py : Training code.

## How to use
### Training
```shell
python main.py

# Default args: training_epoch = 200, z_dim = 20, batch_size = 128, learning_rate = 0.0001
# You can change args: training_epoch = 300, z_dim = 40 batch_size = 64, learning_rate = 0.0005
python main.py --training_epoch 300 --z_dim 40 --batch_size 64 --learning_rate 0.0005
```
