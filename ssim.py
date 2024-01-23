# Loading Libraries
from skimage.metrics import structural_similarity as ssim 
import numpy as np
import matplotlib.pyplot as plt
import cv2         # For Image Processing

import warnings
warnings.filterwarnings("ignore")



# Function for MSE
def mean_squared_error(image1, image2):
    error = np.sum((image1.astype('float') - image2.astype('float'))**2)
    error = error/float(image1.shape[0] * image2.shape[1])
    return error

# Function for image compare
def image_comparison(image1, image2):
    # input image must have the same dimension for comparison
    image2 = cv2.resize(image2,(image1.shape[1::-1]),interpolation=cv2.INTER_AREA)
    m = mean_squared_error(image1, image2)
    s = ssim(image1, image2, multichannel=True)
    print("Mean Squared Error is {}\nStructural Similarity Measure index is: {}".format(m,s))

# load images
image1 = cv2.imread("./input/real.jpg") 
image2 = cv2.imread("./input/tran.jpg")

# display images
def compareImage(originalImage, transformedImage):
    plt.subplot(1, 2, 1)
    plt.imshow(originalImage)
    plt.subplot(1, 2, 2)
    plt.imshow(transformedImage)
    plt.show()


# compare original RGB image similarity
image_comparison(image1,image2)


