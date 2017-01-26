# keras-inceptionV4
Keras implementation of Google's inception v4 model with ported weights!

As described in:
[Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning (Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi)](https://arxiv.org/abs/1602.07261)

Note this Keras implementation tries to follow the [tf.slim definition](https://github.com/tensorflow/models/blob/master/slim/nets/inception_v4.py) as closely as possible.

Pre-Trained weights for this Keras model can be found here: https://github.com/kentsommer/keras-inceptionV4/releases

Once you have downloaded the inception_v4_pretrained.h5, you can evaluate a sample image by performing the following:
* ```$ python evaluate_image.py```
```
Class is: African elephant, Loxodonta africana
Certainty is: 0.938018
```

# Performance Metrics (@Top5, @Top1)
I will be verifying and posting these performance metrics within the next week and a half (by 2/7/2017)
