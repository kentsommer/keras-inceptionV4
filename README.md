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

# News
5/23/2017:

* Enabled support for both Theano and Tensorflow (again... :neckbeard:)
* Added useful training parameters 
  * l2 regularization added to conv layers
  * Variance Scaling initialization added to conv layers
  * Momentum value updated for batch_norm layers
* Updated pre-processing to match paper (subtracts 0.5 instead of 1.0 :fire:)
* Minor code changes and cleanup is also included in the recent changes




# Performance Metrics (@Top5, @Top1)

Error rate on non-blacklisted subset of ILSVRC2012 Validation Dataset (Single Crop):
* Top@1 Error: 19.54%
* Top@5 Error: 4.88%

These error rates are actually slightly lower than the listed error rates in the paper:
* Top@1 Error: 20.0%
* Top@5 Error: 5.0%
