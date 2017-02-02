import tensorflow as tf
slim = tf.contrib.slim
from helper_net.inception_v4 import *
import pickle
import numpy as np

def get_weights():
	checkpoint_file = '../checkpoints/inception_v4.ckpt'
	sess = tf.Session()
	arg_scope = inception_v4_arg_scope()
	input_tensor = tf.placeholder(tf.float32, (None, 299, 299, 3))
	with slim.arg_scope(arg_scope):
	  logits, end_points = inception_v4(input_tensor, is_training=False)
	saver = tf.train.Saver()
	saver.restore(sess, checkpoint_file)

	final_weights = []
	current_bn = []
	final_lr = []

	vars_model = tf.global_variables()
	for i in range(0, len(vars_model), 4):
		for y in range(4):
			key = vars_model[i+y]
			if not "Aux" in key.name:
				if y in [1, 2, 3] and not "Logits" in key.name:
					value = sess.run(key)
					if y == 1:
						current_bn = []
						current_bn.append(np.ones(value.shape[-1]))
						current_bn.append(value)
					elif y == 2:
						current_bn.append(value)
					elif y == 3:
						current_bn.append(value)
						final_weights.append(current_bn)
				elif "Logits" in key.name:
					value = sess.run(key)
					if not "biases" in key.name:
						final_lr.append(value)
					else:
						final_lr.append(value)
						final_weights.append(final_lr)
				else:
					value = sess.run(key)
					final_weights.append([value])
	with open('weights.p', 'wb') as fp:
		pickle.dump(final_weights, fp)	

if __name__ == "__main__":
	get_weights()