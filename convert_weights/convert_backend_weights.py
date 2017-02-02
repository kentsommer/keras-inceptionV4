import sys
sys.path.insert(0,'..')

from keras import backend as K
from keras.utils.np_utils import convert_kernel
import inception_v4
import numpy as np

model = inception_v4.create_model(weights_path="inception_v4_pretrained.h5")

for layer in model.layers:
	if layer.__class__.__name__ in ['Convolution2D']:
		original_w = K.get_value(layer.W)
		print("old_w shape: ", original_w.shape)
		converted_w = convert_kernel(original_w)
		print("new_w shape: ", converted_w.shape)
		K.set_value(layer.W, converted_w)

model.save_weights('../weights/inception_v4_pretrained_theano_tf_dim_order.h5')
