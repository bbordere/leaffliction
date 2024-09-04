import argparse
import pathlib
from sklearn.model_selection import train_test_split
import shutil
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=str,
        help="Augmented Directory path",
    )

    parser.add_argument(
        "dst",
        type=str,
        help="Destination path",
    )

    parser.add_argument(
        "--split", type=int, help="split test part", default=0.2
    )

    args = parser.parse_args()

    train_dir = pathlib.Path(args.dst + "/training")
    test_dir = pathlib.Path(args.dst + "/test")

    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    for root, _, filenames in os.walk(args.path):
        images = [
            pathlib.Path(root + "/" + image)
            for image in filenames
            if image.endswith(".JPG")
        ]

        if len(images) == 0:
            continue

        print(f"Splitting {root} dir...")

        dest_train = pathlib.Path(
            str(train_dir) + "/" + pathlib.Path(root).stem
        )
        if not pathlib.Path.exists(dest_train):
            dest_train.mkdir(parents=True)

        dest_test = pathlib.Path(str(test_dir) + "/" + pathlib.Path(root).stem)
        if not pathlib.Path.exists(dest_test):
            dest_test.mkdir(parents=True)

        train_files, test_files = train_test_split(
            images, test_size=args.split, random_state=42
        )
        for file_path in train_files:
            shutil.copy(file_path, dest_train / file_path.name)
        for file_path in test_files:
            shutil.copy(file_path, dest_test / file_path.name)
    print(f"{args.path} successfully split !")


if __name__ == "__main__":
    main()