
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.stats import entropy
import os
# calculate inception score for cifar-10 in Keras
from math import floor
from numpy import ones
from numpy import expand_dims
from numpy import log
from numpy import mean
from numpy import std
from numpy import exp
from numpy.random import shuffle
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from keras.datasets import cifar10
from skimage.transform import resize
from numpy import asarray

# scale an array of images to a new size
def scale_images(images, new_shape):
	images_list = list()
	for image in images:
		# resize with nearest neighbor interpolation
		new_image = resize(image, new_shape, 0)
		# store
		images_list.append(new_image)
	return asarray(images_list)

# assumes images have any shape and pixels in [0,255]
def calculate_inception_score(images, n_split=10, eps=1E-16):
	# load inception v3 model
	model = InceptionV3()
	# enumerate splits of images/predictions
	scores = list()
	n_part = floor(images.shape[0] / n_split)
	for i in range(n_split):
		# retrieve images
		ix_start, ix_end = i * n_part, (i+1) * n_part
		subset = images[ix_start:ix_end]
		# convert from uint8 to float32
		subset = subset.astype('float32')
		# scale images to the required size
		subset = scale_images(subset, (299,299,3))
		# pre-process images, scale to [-1,1]
		subset = preprocess_input(subset)
		# predict p(y|x)
		p_yx = model.predict(subset)
		# calculate p(y)
		p_y = expand_dims(p_yx.mean(axis=0), 0)
		# calculate KL divergence using log probabilities
		kl_d = p_yx * (log(p_yx + eps) - log(p_y + eps))
		# sum over classes
		sum_kl_d = kl_d.sum(axis=1)
		# average over images
		avg_kl_d = mean(sum_kl_d)
		# undo the log
		is_score = exp(avg_kl_d)
		# store
		scores.append(is_score)
	# average across images
	is_avg, is_std = mean(scores), std(scores)
	return is_avg, is_std

# # load cifar10 images
# (images, _), (_, _) = cifar10.load_data()
# # shuffle images
# shuffle(images)
# print('loaded', images.shape)
# # calculate inception score
# is_avg, is_std = calculate_inception_score(images)
# print('score', is_avg, is_std)


def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(256, 256))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        images.append(img)
    return np.vstack(images)

def get_image_paths(folder_path):
    image_paths = []
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

    for filename in os.listdir(folder_path):
        _, extension = os.path.splitext(filename)
        if extension.lower() in valid_extensions:
            image_paths.append(os.path.join(folder_path, filename))

    return image_paths

# Assuming your generated images are in a folder
def load_generated_images(folder_path):
    image_paths = get_image_paths(folder_path)  # use your get_image_paths function
    images = load_images(image_paths)  # use your load_images function
    # shuffle images
    shuffle(images)
    print('loaded', images.shape)
    return images

# Modify this part of the code
folder_path = "./input/A"  # Path to your generated images
generated_images = load_generated_images(folder_path)

# Calculate the inception score
is_avg, is_std = calculate_inception_score(generated_images)
print('Inception Score:', is_avg, 'Standard Deviation:', is_std)
