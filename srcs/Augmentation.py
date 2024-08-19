import cv2
import argparse
import pathlib
import sys
import matplotlib.pyplot as plt
from cv2.typing import MatLike
from utils import flip, rotate, blur, contrast, scaling, project


def original(img: MatLike) -> MatLike:
    return img


def main():
    parser = argparse.ArgumentParser(
        prog="Augmentation",
        description="Display 6 types of augmentation",
    )
    parser.add_argument("path", help="directory path")
    args = parser.parse_args()

    path = pathlib.Path(args.path)

    if not pathlib.Path.exists(path):
        sys.exit("Path does not exists")

    if not path.name.endswith(".JPG"):
        sys.exit("Not an image")

    img = cv2.cvtColor(cv2.imread(args.path), cv2.COLOR_BGRA2RGBA)

    fig, ax = plt.subplots(1, 7, figsize=(15, 5))

    filters_table = {
        "Original": original,
        "Flip": flip,
        "Rotation": rotate,
        "Blur": blur,
        "Contrast": contrast,
        "Scaling": scaling,
        "Projective": project,
    }
    fig.tight_layout()

    for i, key in enumerate(filters_table.keys()):
        data = filters_table[key](img)
        if key != "Original":
            cv2.imwrite(
                args.path[:-4] + "_" + key + ".JPG",
                cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA),
            )
        ax[i].imshow(data)
        ax[i].axis("off")
        ax[i].set_title(key)

    plt.show()
    # plt.savefig("plots/augmentation.png")


if __name__ == "__main__":
    main()
