import argparse
import pathlib
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys


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
        images = [image for image in filenames if image.endswith(".JPG")]

        if len(images) == 0:
            continue

        files[pathlib.PurePath(root.lower()).name] = len(images)

    if len(files) == 0:
        sys.exit("Path does not contains images")

    palette = sns.color_palette("pastel")
    df = pd.DataFrame(files.items())
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))
    ax[0].pie(x=df[1], colors=palette, autopct="%.1f%%")
    sns.barplot(x=0, y=1, data=df, hue=df[0], ax=ax[1], palette=palette).set(
        xlabel="Labels",
        ylabel="Count",
    )
    plt.xticks(rotation=45)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    fig.suptitle(args.path + " Distribution", fontsize=16)
    plt.show()


if __name__ == "__main__":
    main()
