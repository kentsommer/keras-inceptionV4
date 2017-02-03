import sys
sys.path.insert(0,'..')

from keras import backend as K
from keras.utils.layer_utils import convert_all_kernels_in_model
import inception_v4
import numpy as np
import itertools
import pickle
import re

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

K.set_image_dim_ordering('tf')

th_model = inception_v4.create_model(weights_path="../weights/inception-v4_weights_tf_dim_ordering_tf_kernels.h5")

convert_all_kernels_in_model(th_model)
print("Converted all Kernels")

th_layers = get_layers(th_model)
th_layers = list(itertools.chain.from_iterable(th_layers))

weights = []
for th_layer in th_layers:
	weights.append(th_layer.get_weights())
	print("Saving layer: " + th_layer.name)

pickle.dump(weights, open( "weights_tf_dim_ordering_th_kernels.p", "wb" ) )
print("Saved Pickle of Weights")



