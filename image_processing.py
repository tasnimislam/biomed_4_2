from PIL import Image
import cv2
import numpy as np

def image_preprocessing(path):
    image = Image.open(path)
    image = np.array(image)

    # Preprocessing I have decided
    ad_mean = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    _, otsu = cv2.threshold(ad_mean, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, bin_cann = cv2.threshold(otsu, 127, 255, cv2.THRESH_BINARY)
    return bin_cann / 255.0