from keras import backend as K
import inception_v4
import numpy as np
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 1.0)
    x = np.multiply(x, 2.0)
    return x

# This function comes from Google's ImageNet Preprocessing Script
def central_crop(image, central_fraction):
	"""Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
	if central_fraction <= 0.0 or central_fraction > 1.0:
		raise ValueError('central_fraction must be within (0, 1]')
	if central_fraction == 1.0:
		return image

	img_shape = image.shape
	depth = img_shape[2]
	fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
	bbox_h_start = np.divide(img_shape[0], fraction_offset)
	bbox_w_start = np.divide(img_shape[1], fraction_offset)

	bbox_h_size = img_shape[0] - bbox_h_start * 2
	bbox_w_size = img_shape[1] - bbox_w_start * 2

	image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
	return image

def get_processed_image(img_path):
	# Load image and convert from BGR to RGB
	im = np.asarray(cv2.imread(img_path))[:,:,::-1]
	im = central_crop(im, 0.875)
	im = cv2.resize(im, (299, 299))
	im = preprocess_input(im)
	if K.image_dim_ordering() == "th":
		im = np.transpose(im, (2,0,1))
		im = im.reshape(-1,3,299,299)
	else:
		im = im.reshape(-1,299,299,3)
	return im

if __name__ == "__main__":
	# Create model and load pre-trained weights
	model = inception_v4.create_model(weights='imagenet')

	# Open Class labels dictionary. (human readable label given ID)
	classes = eval(open('validation_utils/class_names.txt', 'r').read())

	# Load test image!
	img_path = 'elephant.jpg'
	img = get_processed_image(img_path)

	# Run prediction on test image
	preds = model.predict(img)
	print("Class is: " + classes[np.argmax(preds)-1])
	print("Certainty is: " + str(preds[0][np.argmax(preds)]))