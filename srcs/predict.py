import multiprocessing.pool
import sys
import cv2
import numpy as np
import argparse
import pathlib

# from Transformation import roi_pcv, blur_pcv, mask_pcv, analyze_pcv, landmarks_pcv
from Transformation import *
from plantcv import plantcv as pcv
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf


import multiprocessing


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

    # if not pathlib.Path(args.img_path).exists():
    #     sys.exit("Path does not exist")

    # if not pathlib.Path(args.img_path).is_file():
    #     sys.exit("Path is not a file")

    # if not args.img_path.endswith((".JPG")):
    #     sys.exit("File is not an image")

    # if not pathlib.Path("my_model_no_filter.keras").exists():
    #     sys.exit("Model does not exist")

    # if not pathlib.Path("class_names.txt").exists():
    #     sys.exit("Class names file does not exist")

    new_model = tf.keras.models.load_model("my_model_no_filter.keras")

    # print(new_model.history)

    class_names = []

    with open("class_names.txt", "r") as file:
        class_names = file.readline().split()

    for root, _, filenames in os.walk(args.img_path):
        images = [root + "/" + image for image in filenames if image.endswith(".JPG")]
        if len(images) == 0:
            continue
        # label = pathlib.Path(root).name
        label = root

        total = len(images)
        good = 0

        for p in images:
            image = cv2.imread(p)
            image = cv2.resize(image, (256, 256))
            gray_img = pcv.rgb2gray_lab(rgb_img=image, channel="a")

            transformations = [
                image,
                roi_pcv(image, gray_img),
                cv2.cvtColor(blur_pcv(image, gray_img), cv2.COLOR_GRAY2RGB),
                mask_pcv(image, gray_img),
                analyze_pcv(image, gray_img),
                landmarks_pcv(image, gray_img),
            ]

            preds = []
            for t in transformations:
                x = np.asarray(t)
                x = np.expand_dims(x, axis=0)
                pred = new_model.predict(x, verbose=0)
                preds.append(class_names[np.argmax(pred)])
            good += preds[np.argmax(preds)] == label
            print(pathlib.Path(p).name, "->", preds[np.argmax(preds)])

            fig, axes = plt.subplots(1, 2, figsize=(8, 6))
            fig.patch.set_facecolor("#1b1b1b")

            axes[0].imshow(transformations[0])
            axes[1].imshow(transformations[3])

            for ax in axes:
                ax.axis("off")

            fig.suptitle(
                "===    DL classification    ===",
                fontsize=30,
                weight="bold",
                y=0.15,
                color="white",
            )

            plt.figtext(
                0.5,
                0.02,
                "Class predicted : " + preds[np.argmax(preds)],
                ha="center",
                fontsize=14,
                color="green",
            )

            if not os.path.exists("predict"):
                os.makedirs("predict")

            if not os.path.exists("predict/" + label):
                os.makedirs("predict/" + label)

            plt.savefig("predict/" + label + "/" + pathlib.Path(p).name + ".png")
            plt.close(fig)


# Grape_spot 0.9813953488372092
# Grape_Black_rot 0.5067911714770797
# Grape_Esca 0.8465991316931982
# Grape_healthy 0.8625592417061612

# Apple_rust 0.12727272727272726
# Apple_healthy 0.5195121951219512
# Apple_Black_rot 0.7887096774193548\
# Apple_scab 0.7042925278219396


if __name__ == "__main__":
    main()
