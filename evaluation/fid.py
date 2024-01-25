import numpy as np
import tensorflow as tf
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from numpy import cov, trace, iscomplexobj, asarray
from numpy.random import shuffle
import glob

# Function to calculate Frechet Inception Distance (FID)
def calculate_fid(model, images1, images2):
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    
    # compute the sum of the mean differences squared
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # compute sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

# Function to preprocess images for InceptionV3
def preprocess_images(image_paths, target_size=(299, 299)):
    images = [img_to_array(load_img(image_path, target_size=target_size)) for image_path in image_paths]
    images = asarray(images)
    images = preprocess_input(images)
    return images

# Function to get a list of image paths from a directory pattern
def get_image_paths(directory_pattern):
    return glob.glob(directory_pattern)

# Paths to your real and generated images
real_image_paths = get_image_paths(".,/input/real/*.jpg")
generated_image_paths = get_image_paths("../input/generated/*.jpg")

# Load and preprocess images
real_images = preprocess_images(real_image_paths)
generated_images = preprocess_images(generated_image_paths)

# Assuming you have less than 2048 images, otherwise, you'd need to batch the computation
shuffle(real_images)
shuffle(generated_images)
real_images = real_images[:min(len(real_images), 2048)]
generated_images = generated_images[:min(len(generated_images), 2048)]

# Load InceptionV3 model
model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))

# Calculate FID
fid = calculate_fid(model, real_images, generated_images)
print(f"FID: {fid}")
