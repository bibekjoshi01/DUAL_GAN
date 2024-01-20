import os
import time
import numpy as np
import tensorflow as tf
from glob import glob
from scipy.stats import entropy
from scipy.linalg import sqrtm
from skimage.transform import resize

# Import the InceptionV3 model for feature extraction
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input

def calculate_fid(real_images, generated_images, inception_model):
    # Resize images to (299, 299) as required by InceptionV3
    real_images_resized = np.array([resize(img, (299, 299)) for img in real_images])
    generated_images_resized = np.array([resize(img, (299, 299)) for img in generated_images])

    # Preprocess images for the InceptionV3 model
    real_images_preprocessed = preprocess_input(real_images_resized)
    generated_images_preprocessed = preprocess_input(generated_images_resized)

    # Get Inception features for real and generated images
    real_features = inception_model.predict(real_images_preprocessed)
    generated_features = inception_model.predict(generated_images_preprocessed)

    # Calculate mean and covariance for real and generated features
    mean_real, cov_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mean_generated, cov_generated = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)

    # Calculate Fréchet distance
    fid = calculate_frechet_distance(mean_real, cov_real, mean_generated, cov_generated)

    return fid

import numpy as np
from scipy.linalg import sqrtm

def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    print("Original sigma1 shape:", sigma1.shape if sigma1 is not None else None)
    print("Original sigma2 shape:", sigma2.shape if sigma2 is not None else None)

    if sigma1 is not None:
        # Ensure covariance matrices are symmetric positive semidefinite
        sigma1 = (sigma1 + sigma1.T) / 2.0

        # Add a small epsilon to the diagonal for numerical stability
        epsilon = 1e-10
        sigma1 += epsilon * np.eye(sigma1.shape[0])

    if sigma2 is not None:
        # Ensure covariance matrices are symmetric positive semidefinite
        sigma2 = (sigma2 + sigma2.T) / 2.0

        # Add a small epsilon to the diagonal for numerical stability
        epsilon = 1e-10
        sigma2 += epsilon * np.eye(sigma2.shape[0])

    print("Adjusted sigma1 shape:", sigma1.shape if sigma1 is not None else None)
    print("Adjusted sigma2 shape:", sigma2.shape if sigma2 is not None else None)

    # Calculate Fréchet distance between two multivariate Gaussians
    term1 = np.sum((mu1 - mu2)**2)
    if sigma1 is not None and sigma2 is not None:
        term2 = np.trace(sigma1 + sigma2 - 2 * sqrtm(np.dot(sigma1, sigma2)))
        return np.sqrt(term1 + term2)
    else:
        return np.nan  # Return a placeholder value if there's an issue with covariance matrices

# Rest of your code...

