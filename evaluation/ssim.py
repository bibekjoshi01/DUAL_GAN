import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os


IMAGE_PATH_PREFIX = "../input/"

# Function for MSE
def mean_squared_error(image1, image2):
    error = np.sum((image1.astype('float') - image2.astype('float'))**2)
    error /= float(image1.shape[0] * image1.shape[1])
    return error

# Function for image comparison
def image_comparison(image1, image2):
    # Ensure images are of the same size
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_AREA)

    m = mean_squared_error(image1, image2)
    s = ssim(image1, image2, multichannel=True)
    return m, s

# Function to get a list of image paths from a directory pattern
def get_image_paths(directory_pattern):
    return glob.glob(directory_pattern)

# Paths to your real and generated images
real_image_paths = get_image_paths(f"{IMAGE_PATH_PREFIX}real/*.jpg")

# Initialize lists to store MSE and SSIM values
mse_values = []
ssim_values = []

IMAGE_COUNTER = 1

# Loop over real images
for real_path in real_image_paths:
    base_name = os.path.splitext(os.path.basename(real_path))[0]  # Extract base name like 'R_1'
    image_number = base_name.split('_')[1]  # Extract number like '1'
    generated_path = f"{IMAGE_PATH_PREFIX}/generated/G_{image_number}.jpg"  # Construct the corresponding generated path

    # Load images and convert to RGB
    image1 = cv2.cvtColor(cv2.imread(real_path), cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(cv2.imread(generated_path), cv2.COLOR_BGR2RGB)

    if image1 is None or image2 is None:
        print(f"Failed to load images for {real_path} or {generated_path}")
        break

    # Compare images and store metrics
    m, s = image_comparison(image1, image2)
    mse_values.append(m)
    ssim_values.append(s)
    print(f"Image {IMAGE_COUNTER} - MSE: {m}, SSIM: {s}")
    IMAGE_COUNTER += 1

# Calculate and print the mean MSE and SSIM
mean_mse = np.mean(mse_values)
mean_ssim = np.mean(ssim_values)
print(f"\nOverall Mean MSE: {mean_mse}\nOverall Mean SSIM: {mean_ssim}")

# Plotting the MSE and SSIM values
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(mse_values, label='MSE')
plt.title("MSE of Image Pairs")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(ssim_values, label='SSIM')
plt.title("SSIM of Image Pairs")
plt.legend()

plt.show()
