from keras.utils.visualize_util import plot
import inception_v4

if __name__ == "__main__":
	model = inception_v4.create_model()
	plot(model, to_file="inception_v4.png", show_shapes=True)