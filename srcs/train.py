import os
import tensorflow as tf
import argparse
import pathlib
import sys
from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np
from tensorflow.keras.preprocessing import image

IMAGE_SIZE = 128


def predict_on_folder(
    model: any,
    folder_path: str,
    img_height: int,
    img_width: int,
    class_names: list[str],
) -> np.ndarray:
    """make predictions on folder

    Args:
        model (any): model to use for predictions
        folder_path (str): path to image
        img_height (int): height of images
        img_width (int): width of images
        class_names (list[str]): class names

    Returns:
        np.ndarray: predictions
    """
    results = []
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = image.load_img(img_path, target_size=(img_height, img_width))
        img_array = image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        predictions = model.predict(img_array)
        score = predictions
        print(score)

        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        results.append((img_name, predicted_class, confidence))

    return results


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
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        args.data_dir,
        validation_split=0.2,
        subset="validation",
        seed=42,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=args.batch_size,
    )

    class_names = train_ds.class_names

    with open("class_names.txt", "w") as file:
        print(*class_names, file=file)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
    )

    model = models.Sequential(
        [
            layers.Rescaling(1.0 / 255),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.1),
            layers.Flatten(),
            layers.Dense(512, activation="relu"),
            layers.Dense(len(train_ds.class_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[early_stopping],
    )

    model.save("model.keras")


if __name__ == "__main__":
    main()
