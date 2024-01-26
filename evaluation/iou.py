import os
import numpy as np
from PIL import Image

def calculate_iou(real_images, generated_images, threshold=0.5):
    real_images_binary = (real_images > threshold).astype(np.uint8)
    generated_images_binary = (generated_images > threshold).astype(np.uint8)

    intersection = np.logical_and(real_images_binary, generated_images_binary)
    union = np.logical_or(real_images_binary, generated_images_binary)
    iou = np.sum(intersection) / np.sum(union)

    print(f"Threshold: {threshold}")
    print(f"Intersection: {np.sum(intersection)}")
    print(f"Union: {np.sum(union)}")
    print(f"IoU: {iou}")

    return iou


def print_image_statistics(images, name):
    print(f"{name} Image Statistics:")
    print(f"  Min Pixel Value: {np.min(images)}")
    print(f"  Max Pixel Value: {np.max(images)}")
    print(f"  Mean Pixel Value: {np.mean(images)}")
    print(f"  Std Dev Pixel Value: {np.std(images)}")

def load_images(filenames):
    images = [np.array(Image.open(filename)) for filename in filenames]
    return np.array(images)

def test():
    real_images_A_filenames = [os.path.join("./eval/A/real", filename) for filename in os.listdir("./eval/A/real")]
    generated_images_A_filenames = [os.path.join("./eval/A/generated", filename) for filename in os.listdir("./eval/A/generated")]

    real_images_B_filenames = [os.path.join("./eval/B/real", filename) for filename in os.listdir("./eval/B/real")]
    generated_images_B_filenames = [os.path.join("./eval/B/generated", filename) for filename in os.listdir("./eval/B/generated")]

    real_images_A = load_images(real_images_A_filenames)
    generated_images_A = load_images(generated_images_A_filenames)

    real_images_B = load_images(real_images_B_filenames)
    generated_images_B = load_images(generated_images_B_filenames)

    real_images_A_filenames.sort()
    generated_images_A_filenames.sort()

    real_images_B_filenames.sort()
    generated_images_B_filenames.sort()

    for real_A, generated_A in zip(real_images_A_filenames, generated_images_A_filenames):
        print(f"Pair A: {real_A}, {generated_A}")

    for real_B, generated_B in zip(real_images_B_filenames, generated_images_B_filenames):
        print(f"Pair B: {real_B}, {generated_B}")

    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]

    for threshold in thresholds:
        iou_A = calculate_iou(real_images_A, generated_images_A, threshold)
        iou_B = calculate_iou(real_images_B, generated_images_B, threshold)

        print(f"IoU for domain A with threshold {threshold}: {iou_A}")
        print(f"IoU for domain B with threshold {threshold}: {iou_B}")

if __name__ == "__main__":
    test()
