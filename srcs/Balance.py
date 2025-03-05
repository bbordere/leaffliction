import pathlib
import argparse
import sys
import cv2
from utils import filters_table, extensions
import os
from shutil import copytree
from alive_progress import alive_bar


def apply_augmentation(path: str, counter: int, bar) -> int:
    """apply augmentation on file to create new images to balance dataset

    Args:
        path (str): path to image
        counter (int): files number limit
        bar (_type_): progress bar instance

    Returns:
        int: files number counter after augmentation
    """
    for ext in extensions:
        if ext in path:
            return counter
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGRA2RGBA)
    for key in filters_table:
        data = filters_table[key](img)
        cv2.imwrite(
            path[:-4] + "_" + key + ".JPG",
            cv2.cvtColor(data, cv2.COLOR_RGBA2BGRA),
        )
        bar()
        counter -= 1
        if counter == 0:
            break
    return counter


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Distribution",
        description="Show distribution",
    )
    parser.add_argument("path", help="directory path")
    args = parser.parse_args()

    path = pathlib.Path(args.path)

    if not pathlib.Path.exists(path):
        sys.exit("Path does not exists")

    if not pathlib.Path.is_dir(path):
        sys.exit("Path is not a directory")

    files = {}
    for root, _, filenames in os.walk(args.path):
        images = [
            root + "/" + image for image in filenames if image.endswith(".JPG")
        ]
        if len(images) == 0:
            continue
        files[pathlib.PurePath(root.lower()).name] = images

    label_max = max(files, key=lambda key: len(files[key]))

    target_num_img = len(files[label_max])

    files.pop(label_max)

    for label in files:
        counter = target_num_img - len(files[label])
        with alive_bar(counter) as bar:
            for file in files[label]:
                counter = apply_augmentation(file, counter, bar)
                if counter == 0:
                    break

    copytree(args.path, "./augmented_directory", dirs_exist_ok=True)


if __name__ == "__main__":
    main()
