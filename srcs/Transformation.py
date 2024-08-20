import argparse
from plantcv import plantcv as pcv
import cv2
import pathlib
import sys
import os
from cv2.typing import MatLike


def analyse_pcv(img: MatLike) -> MatLike:
    gray_image = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    bin_mask = pcv.threshold.binary(
        gray_img=gray_image, threshold=100, object_type="light"
    )

    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=255, w=255)

    mask = pcv.roi.filter(mask=bin_mask, roi=roi, roi_type="partial")

    shape_img = pcv.analyze.size(img=img, labeled_mask=mask, n_labels=1)
    pcv.plot_image(shape_img)


def roi_pcv(img: MatLike) -> MatLike:
    gray_image = pcv.rgb2gray_hsv(rgb_img=img, channel="s")
    bin_img = pcv.invert(
        pcv.threshold.binary(gray_img=gray_image, threshold=100, object_type="light")
    )
    pcv.plot_image(bin_img)


def blur_pcv(img: MatLike) -> MatLike:
    gray_image = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    bin_img = pcv.threshold.otsu(gray_img=gray_image, object_type="dark")
    blur = pcv.gaussian_blur(img=bin_img, ksize=(9, 9), sigma_x=0, sigma_y=None)
    pcv.plot_image(blur)
    return blur


def mask_pcv(img: MatLike) -> MatLike:
    cs = pcv.visualize.colorspaces(rgb_img=img, original_img=False)
    gray_image = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    background = pcv.rgb2gray_lab(rgb_img=img, channel="a")

    mask = pcv.threshold.otsu(gray_img=gray_image, object_type="dark")
    mask_back = pcv.threshold.otsu(gray_img=background, object_type="dark")

    masked_image = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    masked_image = pcv.apply_mask(img=img, mask=mask_back, mask_color="white")
    pcv.plot_image(masked_image)


def apply_filter(path: str, filter: str) -> None:
    img = cv2.imread(path)
    match filter:
        case "blur":
            blur_pcv(img)
        case "mask":
            mask_pcv(img)
        case "roi":
            roi_pcv(img)
        case "analyze":
            analyse_pcv(img)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Transformation",
        description="Apply transformation on image or folder",
    )
    parser.add_argument("src", help="path to image or directory")
    parser.add_argument("-d", "--destination", help="path to image or directory")
    parser.add_argument(
        "-t",
        "--transformation",
        help="trans",
        default="all",
        choices=["all", "blur", "mask", "roi", "analyze", "landmarks", "intensity"],
    )

    args = parser.parse_args()

    path = pathlib.Path(args.src)

    if not pathlib.Path.exists(path):
        sys.exit("Path does not exists")

    if pathlib.Path.is_dir(path):
        print("DIR")
    else:
        print("FILE")
        apply_filter(args.src, args.transformation)


if __name__ == "__main__":
    main()
