# Bag-of-Features Pooling for Deep Convolutional Neural Networks

This implementation is based on the implementation used for conducting the experiments in the [Bag-of-Features Pooling method for Deep Convolutional Neural Networks](https://arxiv.org/abs/1707.08105) paper. This implementation is slower than the *lasagne*-based implementation that we provide in the [main repository](). However, it is also more flexible, e.g., it allows for using separate codebooks for each spatial region.

Note that the obtained results might slightly vary due to the non-deterministic behaviour of the libraries (CUDA) used for the GPU calculations and the clustering algorithm used for the initialization of the codebook. To avoid these issues we explicitly avoid using non-determining algorithms during the optimization in the results reported here. To do so, you can add the following in the *theano.rc* configuration file:

<pre>
[dnn.conv]
algo_bwd_filter=deterministic
algo_bwd_data=deterministic
</pre>

After using this configuration and fixing the seeds, the following results should be obtained:


| Model         | 28 x 28 | 20 x 20 | 
| ------------- | --------- | ---------   |
| CNN           | 0.56 %    |  -     |
| GMP           | 0.78 %    | 3.31 %      |
| SPP           | 0.55 %    | 1.49 %      |
| CBoF (64, 1)   | **0.47 %**    | **0.99 %** |


If you use this code in your work please cite the following paper:

<pre>
@InProceedings{cbof_iccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Bag-of-Features Pooling for Deep Convolutional Neural Networks},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision (to appear)},
year = {2017}
}
</pre>

