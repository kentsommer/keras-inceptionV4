from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.insert(0,'..')

import tensorflow as tf
import numpy as np
import itertools
import pickle
import os
import re

import inception_v4

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(myobject):
    return [ atoi(c) for c in re.split('(\d+)', myobject.name) ]

def setWeights(layers, weights):
	for index, layer in enumerate(layers):
		layer.set_weights(weights[index])
		print(layer.name + " weights have been set!")
	print("Finished Setting Weights!")

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


if __name__ == "__main__":
	model = inception_v4.create_model()

	with open('weights.p', 'rb') as fp:
		weights = pickle.load(fp)

	# Get layers to set
	layers = get_layers(model)
	layers = list(itertools.chain.from_iterable(layers))

	# Set the layer weights
	setWeights(layers, weights)

	# Save model weights in h5 format
	model.save_weights("../weights/inception-v4_weights_tf_dim_ordering_tf_kernels.h5")
	print("Finished saving weights in h5 format")


	
