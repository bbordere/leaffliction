import tensorflow as tf

import cv2
import sys
import numpy as np

from Transformation import *
from plantcv import plantcv as pcv


from tensorflow.keras import datasets, layers, models

new_model = tf.keras.models.load_model("my_model_no_filter.keras")

class_names = []

with open("class_names.txt", "r") as file:
    class_names = file.readline().split()

image = cv2.imread(sys.argv[1])
image = cv2.resize(image, (256, 256))
gray_img = pcv.rgb2gray_lab(rgb_img=image, channel="a")

transformations = [
    image,
    roi_pcv(image, gray_img),
    blur_pcv(image, gray_img),
    mask_pcv(image, gray_img),
    analyze_pcv(image, gray_img),
    landmarks_pcv(image, gray_img),
]

for t in transformations:
    x = np.asarray(image)
    x = np.expand_dims(x, axis=0)
    pred = new_model.predict(x)
    print(class_names[np.argmax(pred)])
