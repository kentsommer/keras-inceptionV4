import sys
sys.path.insert(0,'..')

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
import inception_v4
import numpy as np
import itertools
import pickle
import os
import re

def shuffle_rows(original_w):
	converted_w = np.zeros(original_w.shape)
	count = 0
	for index, row in enumerate(original_w):
		if (index % 256) == 0 and index != 0:
			count += 1
		new_index = ((index % 256) * 6) + count
		print("index from " + str(index) + " -> " + str(new_index))
		converted_w[new_index] = row
		initial = 1
	return converted_w

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(myobject):
    return [ atoi(c) for c in re.split('(\d+)', myobject.name) ]

def get_layers(model):
	# Get Trainable layers
	layers = model.layers
	layers.sort(key=natural_keys)
	result = []
	for i in range(len(layers)):
		try:
			layer = model.layers[i]
			if layer.trainable:
				bad = ["pooling", "flatten", "dropout", "activation"]
				if not any(word in layer.name for word in bad):
					result.append(layer)
		except:
			continue
	bn,cv,fn=result[:int((len(result)-1)/2)],result[int((len(result)-1)/2):],result[-1]
	res_zipped = zip(cv, bn)
	out_prep = [list(elem) for elem in res_zipped]
	out = out_prep + [[fn]]
	return out


K.set_image_dim_ordering('th')

th_model = inception_v4.create_model()

th_layers = get_layers(th_model)
th_layers = list(itertools.chain.from_iterable(th_layers))

conv_classes = {
        'Convolution1D',
        'Convolution2D',
        'Convolution3D',
        'AtrousConvolution2D',
        'Deconvolution2D',
    }

weights_list = pickle.load( open( "weights_tf_dim_ordering_th_kernels.p", "rb" ) )

for index, th_layer in enumerate(th_layers):
	if th_layer.__class__.__name__ in conv_classes:
		weights = weights_list[index]
		weights[0] = weights[0].transpose((3,2,0,1))
		th_layer.set_weights(weights)
		print('converted ', th_layer.name)
	# elif th_layer.__class__.__name__ == "Dense":
	# 	weights = weights_list[index]
	# 	weights[0] = shuffle_rows(weights[0])
	# 	th_layer.set_weights(weights)
	# 	print('converted ', th_layer.name)
	else:
		th_layer.set_weights(weights_list[index])
		print('Set: ', th_layer.name)


th_model.save_weights('../weights/inception-v4_weights_th_dim_ordering_th_kernels.h5')


