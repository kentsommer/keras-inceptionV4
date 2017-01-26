from PIL import Image
import inception_v4
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

def preprocess_input(x):
    x = np.divide(x, 255.0)
    x = np.subtract(x, 1.0)
    x = np.multiply(x, 2.0)
    return x

if __name__ == "__main__":
	model = inception_v4.create_model(weights_path="inception_v4_pretrained.h5")

	classes = eval(open('classes.txt', 'r').read())

	img_path = 'elephant.jpg'
	im = Image.open(img_path).resize((299,299))
	im = np.array(im)
	im = preprocess_input(im)
	im = im.reshape(-1,299,299,3)

	preds = model.predict(im)
	print("Class is: " + classes[np.argmax(preds)-1])
	print("Certainty is: " + str(preds[0][np.argmax(preds)]))
