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

## Result
### Reconstruction

![Alt Text](https://github.com/DevKiHyun/CVAE-Tensorflow/blob/master/result/reconstruction.png)

### Generation

| Give labels  | 0, 1, 2, 3, 4, 9, 8, 7 |
| ------------- | ------------- |
| Generation  | ![Alt Text](https://github.com/DevKiHyun/CVAE-Tensorflow/blob/master/result/generation.png)  |

### 2D-manifold

![Alt Text](https://github.com/DevKiHyun/CVAE-Tensorflow/blob/master/result/manifold.png)

### 2D-manifold walking

![Alt Text](https://github.com/DevKiHyun/CVAE-Tensorflow/blob/master/result/walking.png)

## Reference
[오토인코더의 모든것('All of autoencoder')](https://www.youtube.com/watch?v=o_peo6U7IRM&feature=youtu.be)

[Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908)
