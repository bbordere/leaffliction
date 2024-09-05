import sys
import cv2
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn
import pathlib
from alive_progress import alive_bar
import os

import tensorflow as tf
from Transformation import ImageTransformer
import matplotlib.pyplot as plt

class_names = []
truth = []
predict = []

IMAGE_SIZE = None


def label_name(class_names: list, path: str) -> str:
    """get name of label of path

    Args:
        class_names (list): class_names possibilities
        path (str): path

    Returns:
        str: label name
    """
    return next((c for c in class_names if c in str(path)), None)


def make_prediction(
    path: str, model: any, verbose: bool = False, bar: any = None
):
    """make prediction for given path with give model

    Args:
        path (str): path to image to predict
        model (any): model instance to use
        verbose (bool, optional): print prediction score. Defaults to False.
        bar (any, optional): bar instance for directory mode. Defaults to None.
    """
    t = ImageTransformer()
    t.open(path, IMAGE_SIZE, pred=True)
    transformations = np.stack(
        [
            t.img,
            t.roi(),
            cv2.cvtColor(t.gaussian_blur(), cv2.COLOR_GRAY2RGB),
            t.mask(),
            t.analyze(),
            t.landmarks(),
            cv2.cvtColor(t.canny_edges(), cv2.COLOR_GRAY2RGB),
        ]
    )
    predictions = np.mean(
        model.predict(transformations, verbose=False), axis=0
    )

    predicted_class = class_names[np.argmax(predictions)]
    truth.append(labels[label_name(class_names, path)])
    predict.append(np.argmax(predictions))
    if verbose:
        print(path, predicted_class, f"{100 * np.max(predictions)}%")
    if bar:
        bar()
    display_prediction(path, t, predicted_class, bar)


def handle_dir(path: str, model: any):
    """handle directory for predictions

    Args:
        path (str): path to directory
        model (any): instance of model to use
    """
    for root, _, filenames in os.walk(path):
        images = [
            root + "/" + image for image in filenames if image.endswith(".JPG")
        ]
        if len(images) == 0:
            continue

        with alive_bar(len(images), title=root) as bar:
            for p in images:
                make_prediction(p, model, False, bar)


def display_prediction(
    path: str, t: ImageTransformer, label_pred, save: bool = False
):
    """display prediction

    Args:
        path (str): path to image
        t (ImageTransformer): image transformer class
        label_pred (_type_): label predicted
        save (bool, optional): save to disk instead of displaying. Defaults to False.
    """
    size = t.original_img.shape[:2]
    t.open(path, size)

    fig, axes = plt.subplots(1, 2, figsize=(8, 6))
    fig.patch.set_facecolor("#1b1b1b")

    axes[0].imshow(cv2.cvtColor(t.img, cv2.COLOR_RGB2BGR))
    axes[1].imshow(cv2.cvtColor(t.roi(), cv2.COLOR_BGR2RGB))

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
        "Class predicted : " + label_pred,
        ha="center",
        fontsize=14,
        color="green",
    )

    if not save:
        plt.show()
        plt.close()
        return

    dest = pathlib.Path("predict/" + path.replace(pathlib.Path(path).name, ""))
    if not dest.exists():
        dest.mkdir(parents=True)

    plt.savefig(str(dest) + "/" + pathlib.Path(path).name + ".png")
    plt.close(fig)


def main():
    global class_names
    global labels
    global truth
    global predict
    global IMAGE_SIZE

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Path to the image",
    )

    parser.add_argument(
        "model_path",
        type=str,
        default="model.keras",
        help="Path to the model",
    )

    parser.add_argument(
        "-c",
        "--c_mat",
        default=False,
        action="store_true",
        help="Save the confusion matrix",
    )

    args = parser.parse_args()

    if not pathlib.Path(args.path).exists():
        sys.exit("Path does not exist")

    if not pathlib.Path(args.model_path).exists():
        sys.exit("Model does not exist")

    if not pathlib.Path("class_names.txt").exists():
        sys.exit("Class names file does not exist")

    model = tf.keras.models.load_model(args.model_path)
    print(model.summary())

    config = model.get_config()
    IMAGE_SIZE = (
        config["layers"][0]["config"]["batch_shape"][1],
        config["layers"][0]["config"]["batch_shape"][2],
    )

    with open("class_names.txt", "r") as file:
        class_names = file.readline().split()
    labels = {l: i for i, l in enumerate(class_names)}

    if pathlib.Path.is_dir(pathlib.Path(args.path)):
        handle_dir(args.path, model)
        print(
            f"Accuracy on directory: \
{accuracy_score(truth, predict) * 100:.4f}%"
        )
    else:
        make_prediction(args.path, model)

    if args.c_mat:
        class_predicted = [class_names[c] for c in set(predict)]
        cf_matrix = confusion_matrix(truth, predict)
        seaborn.heatmap(
            cf_matrix,
            annot=True,
            cmap="Blues",
            xticklabels=class_predicted,
            yticklabels=class_predicted,
            fmt="g",
        )
        plt.xticks(rotation=45, ha="right")
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion matrix of " + args.path)
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
