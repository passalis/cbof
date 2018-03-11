# Bag-of-Features Pooling for Deep Convolutional Neural Networks

**IMPORTANT: Given the uncertain future of theano, we also provide a [*keras*-based implementation](https://github.com/passalis/keras_cbof) of the proposed method.**

In this repository we provide an efficient and simple re-implementation of the [Bag-of-Features Pooling method for Deep Convolutional Neural Networks](https://arxiv.org/abs/1707.08105) using the Lasagne framework. The provided lasagne layer can be used in any lasagne-based model. The distance between the extracted feature vectors and the codebook is calculated using convolutional layers (exploiting that the squared distance ||x-y||^2 can be calculated using three inner products, i.e., x^2+y^2-2xy), significantly speeding up the training/testing speed.

We provide an example of using the proposed method in mnist_example.py and we compare the BoF pooling to the plain SPP polling. The proposed method can both increase the classification perfomance and provide better scale invariance, as shown below (the classification error on the MNIST dataset is reported):


| Model         | Scale = 1 | Scale = 0.8 |  Scale = 0.7 | 
| ------------- | --------- | ---------   | ---------    | 
| SPP           | 0.68 %    | 4.08 %      | 36.78 %      |
| BoF Pooling   | **0.54 %**    | **1.40 %**      | **17.60 %**    |

Note that this is not the implementation used for conducting the experiments in our [paper](https://arxiv.org/abs/1707.08105). The original (slower, but more flexible) implementation can be found in [cbof_paper](cbof_paper).

If you use this code in your work please cite the following paper:

<pre>
@InProceedings{cbof_iccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Bag-of-Features Pooling for Deep Convolutional Neural Networks},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision},
year = {2017}
}
</pre>

Check my [website](http://users.auth.gr/passalis) for more projects and stuff!
