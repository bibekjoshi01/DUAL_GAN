import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from scipy.stats import entropy
import os

def load_images(image_paths):
    images = []
    for path in image_paths:
        img = image.load_img(path, target_size=(299, 299))
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


def calculate_inception_score(generated_image_paths, num_splits=10):
    inception_model = InceptionV3(weights='imagenet', include_top=True)  # include_top=True

    def get_inception_probs(images):
        preds = inception_model(images, training=False)
        return preds

    generated_images = load_images(generated_image_paths)
    scores = []
    splits = np.array_split(generated_images, num_splits)

    for split in splits:
        incep_probs = get_inception_probs(tf.convert_to_tensor(split, dtype=tf.float32))
        if not np.any(incep_probs):  # Check if incep_probs is empty
            continue

        p_y = np.mean(incep_probs, axis=0)
        kl_divs = []

        for i in range(len(split)):
            kl_div = entropy(incep_probs[i], p_y)
            kl_divs.append(kl_div)

        avg_kl_div = np.mean(kl_divs)
        scores.append(np.exp(avg_kl_div))

        print("Individual KL Divergences:", kl_divs)

    if not scores:
        print("Unable to calculate Inception Score. Please check your generated images.")
        return np.nan

    inception_score = np.mean(scores)
    return inception_score

# Example usage:
folder_path = "./datasets/sketch-photo/val/A"  # Replace with the path to your folder containing images
generated_image_paths = get_image_paths(folder_path)

inception_score = calculate_inception_score(generated_image_paths)
print("Inception Score:", inception_score)