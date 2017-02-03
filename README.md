# Keras Inception-V4
Keras implementation of Google's inception v4 model with ported weights!

As described in:
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi)](https://arxiv.org/abs/1602.07261)

Note this Keras implementation tries to follow the [tf.slim definition](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v4.py) as closely as possible.

Pre-Trained weights for this Keras model can be found here (ported from the tf.slim ckpt): https://github.com/kentsommer/keras-inceptionV4/releases

You can evaluate a sample image by performing the following (weights are downloaded automatically):
* ```$ python evaluate_image.py```
```
Loaded Model Weights!
Class is: African elephant, Loxodonta africana
Certainty is: 0.868498
```

# Performance Metrics (@Top5, @Top1)

Error rate on non-blacklisted subset of ILSVRC2012 Validation Dataset (Single Crop):
* Top@1 Error: 19.54%
* Top@5 Error: 4.88%

These error rates are actually slightly lower than the listed error rates in the paper:
* Top@1 Error: 20.0%
* Top@5 Error: 5.0%

# News
2/3/2017:

1. This now fully supports both the Theano and Tensorflow backends! This means you can use whatever dim ordering you like as well as picking the backend you prefer! All of the following will output the same thing:
  * tf_dim + Tensorflow
  * th_dim + Tensorflow
  * th_dim + Theano 
  * tf_dim + Theano
  
2. Weights no longer have to be downloaded manually! Simply run the evaluate script and the correct weights will be downloaded automatically!
