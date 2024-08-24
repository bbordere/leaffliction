import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import argparse
import pathlib
import sys

IMAGE_SIZE = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="?",
        default="path_to_dataset/",
        help="Path to the data directory",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="Number of epochs",
    )
    args = parser.parse_args()

    path = pathlib.Path(args.data_dir)

    if not path.exists():
        sys.exit("Path does not exist")
    if not path.is_dir():
        sys.exit("Path is not a directory")

    for file in path.iterdir():
        if file.is_file():
            if not file.name.endswith((".JPG")):
                sys.exit("Directory contains non-image files")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names

    with open("class_names.txt", "w") as file:
        print(*class_names, file=file)

    # AUTOTUNE = tf.data.AUTOTUNE

    num_classes = 8

    # model = tf.keras.Sequential(
    #     [
    #         tf.keras.layers.Rescaling(1.0 / 255),
    #         tf.keras.layers.Conv2D(16, 3, activation="relu"),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Conv2D(32, 3, activation="relu"),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Conv2D(64, 3, activation="relu"),
    #         tf.keras.layers.MaxPooling2D(),
    #         tf.keras.layers.Flatten(),
    #         tf.keras.layers.Dense(128, activation="relu"),
    #         tf.keras.layers.Dense(64, activation="relu"),
    #         tf.keras.layers.Dense(num_classes, activation="softmax"),
    #     ]
    # )

    # model.compile(
    #     optimizer="adam",
    #     loss=tf.losses.SparseCategoricalCrossentropy(),
    #     metrics=["accuracy"],
    # )

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Rescaling(1.0 / 255))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(16, (3, 3), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(16, (1, 1), activation="relu"))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(256, activation="relu"))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs)

    model.save("my_model_test2.keras")


if __name__ == "__main__":
    main()
