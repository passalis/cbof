# Bag-of-Features Pooling for Deep Convolutional Neural Networks

This is the implementation used for conducting the experiments in the [Bag-of-Features Pooling method for Deep Convolutional Neural Networks]() paper. This implementation is slower than the *lasagne*-based implementation that we provide in the [main repository](). However, it is also more flexibly, i.e., it allows for using separate codebooks for each spatial region.

Note that the obtain results might slightly vary due to the non-deterministic behaviour of the libraries (CUDA) used for the GPU calculations. To avoid these issues we explicitly avoiding using non-determining algorithms during the optimization. To do so, you can add the following in the *theano.rc* configuration file:

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
| ------------- | --------- | ---------  |
| CBoF (64, 1)   | ** ? %**    | **? %** |


If you use this code in your work please cite the following paper:

<pre>
@InProceedings{cbof_iccv,
author = {Passalis, Nikolaos and Tefas, Anastasios},
title = {Ask Your Neurons: A Neural-Based Approach to Answering Questions About Images},
booktitle = {Proceedings of the IEEE International Conference on Computer Vision (to appear)},
year = {2017}
}
</pre>

