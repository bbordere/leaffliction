import tensorflow as tf
import sys
import cv2
import numpy as np
import argparse
import pathlib

from Transformation import roi_pcv, blur_pcv, mask_pcv, analyze_pcv, landmarks_pcv
from plantcv import plantcv as pcv


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "img_path",
        type=str,
        nargs="?",
        default="path_to_image/",
        help="Path to the image",
    )

    args = parser.parse_args()

    if not pathlib.Path(args.img_path).exists():
        sys.exit("Path does not exist")

    if not pathlib.Path(args.img_path).is_file():
        sys.exit("Path is not a file")

    if not args.img_path.endswith((".JPG")):
        sys.exit("File is not an image")

    if not pathlib.Path("my_model_no_filter.keras").exists():
        sys.exit("Model does not exist")

    if not pathlib.Path("class_names.txt").exists():
        sys.exit("Class names file does not exist")

    new_model = tf.keras.models.load_model("my_model.keras")

    class_names = []

    with open("class_names.txt", "r") as file:
        class_names = file.readline().split()

    image = cv2.imread(args.img_path)
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


if __name__ == "__main__":
    main()
