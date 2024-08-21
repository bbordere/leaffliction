import argparse
from plantcv import plantcv as pcv
import cv2
import pathlib
import sys
from cv2.typing import MatLike
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.gridspec as grsp
from utils import params

pcv.params.line_thickness = 2
pcv.params.dpi = 100


def analyze_pcv(img: MatLike, gray_img: MatLike) -> MatLike:
    bin_mask = pcv.threshold.otsu(gray_img=gray_img, object_type="dark")
    roi = pcv.roi.rectangle(img=img, x=0, y=0, h=255, w=255)

    mask = pcv.roi.filter(mask=bin_mask, roi=roi, roi_type="partial")

    shape_img = pcv.analyze.size(img=img, labeled_mask=mask)
    return shape_img


def roi_pcv(img: MatLike, gray_img: MatLike) -> MatLike:
    bin_mask = pcv.threshold.triangle(gray_img=gray_img, object_type="dark")

    color_pix = np.where(bin_mask != 0)
    x, y, w, h = (
        np.min(color_pix[1]),
        np.min(color_pix[0]),
        np.max(color_pix[1]) - np.min(color_pix[1]),
        np.max(color_pix[0]) - np.min(color_pix[0]),
    )
    roi = pcv.roi.rectangle(img, x, y, h, w)

    mask = pcv.roi.filter(mask=bin_mask, roi=roi, roi_type="partial")
    colorized_mask = pcv.visualize.colorize_masks(masks=[mask], colors=["green"])
    merged_image = pcv.visualize.overlay_two_imgs(colorized_mask, img, alpha=0.4)
    cv2.rectangle(merged_image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return merged_image


def landmarks_pcv(img: MatLike, gray_img: MatLike) -> MatLike:
    res = img.copy()
    bin_img = pcv.threshold.otsu(gray_img=gray_img, object_type="dark")
    bin_img = pcv.fill_holes(bin_img)
    left, right, center = pcv.homology.y_axis_pseudolandmarks(img=img, mask=bin_img)

    for pl, pc, pr in zip(left.astype(int), right.astype(int), center.astype(int)):
        cv2.circle(res, pl[0], radius=3, color=(209, 23, 206), thickness=-1)
        cv2.circle(res, pr[0], radius=3, color=(49, 14, 204), thickness=-1)
        cv2.circle(res, pc[0], radius=3, color=(222, 86, 18), thickness=-1)
    return res


def save_histogram(channels: list, params: dict, src: pathlib.Path, dest: pathlib.Path):
    fig = plt.figure(figsize=(10, 10))
    new_plot = fig.add_subplot()
    for i, key in enumerate(params):
        hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
        hist = hist / hist.sum() * 100
        new_plot.plot(hist, color=params[key], label=key)

    new_plot.set_axis_off()
    new_plot.figure.tight_layout()
    new_plot.figure.savefig(
        str(dest) + "/" + src.name[:-4] + "_trans_histogram.JPG", dpi=25.6
    )
    plt.close()


def color_hist_pcv(
    img: MatLike,
    ax: np.ndarray = None,
    save: bool = False,
    src: pathlib.Path = None,
    dest: pathlib.Path = None,
):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    channels = [
        img[:, :, 0],
        img[:, :, 1],
        img[:, :, 2],
        hsv[:, :, 0],
        hsv[:, :, 1],
        hsv[:, :, 2],
        lab[:, :, 0],
        lab[:, :, 1],
        lab[:, :, 2],
    ]

    if save:
        save_histogram(channels, params, src, dest)
        return

    if ax is not None:
        for i, key in enumerate(params):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            hist = hist / hist.sum() * 100
            ax.plot(hist, color=params[key], label=key)
            ax.set_xlim([0, 256])
            ax.set_xticks(range(0, 255, 25))

        ax.grid(zorder=0)
        ax.set_title("Color Histograms (Percentage of Pixels)")
        ax.set_xlabel("Intensity")
        ax.set_ylabel("Percentage of Pixels")
        ax.legend()

    else:
        for i, key in enumerate(params):
            hist = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
            hist = hist / hist.sum() * 100
            plt.plot(hist, color=params[key], label=key)
            plt.xlim([0, 256])
            plt.xticks(range(0, 255, 25))

        plt.grid(zorder=0)
        plt.title("Color Histograms (Percentage of Pixels)")
        plt.xlabel("Intensity")
        plt.ylabel("Percentage of Pixels")
        plt.legend()
        plt.show()


def handle_all(
    img: MatLike,
    gray_img: MatLike,
    path: pathlib.Path,
    dest: pathlib.Path,
    save: bool = False,
) -> None:
    fig = plt.figure(figsize=(10, 6))
    gs = grsp.GridSpec(2, 6, height_ratios=[2, 1])

    transformations = {
        "Original": cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
        "Blur": cv2.cvtColor(blur_pcv(gray_img), cv2.COLOR_RGB2BGR),
        "Mask": cv2.cvtColor(mask_pcv(img, gray_img), cv2.COLOR_RGB2BGR),
        "ROI": cv2.cvtColor(roi_pcv(img, gray_img), cv2.COLOR_RGB2BGR),
        "Analyze": cv2.cvtColor(analyze_pcv(img, gray_img), cv2.COLOR_RGB2BGR),
        "Pseudo-Landmarks": cv2.cvtColor(
            landmarks_pcv(img, gray_img), cv2.COLOR_RGB2BGR
        ),
    }

    for i, key in enumerate(transformations):
        if save:
            if key == "Original":
                continue
            cv2.imwrite(
                str(dest) + "/" + path.name[:-4] + "_trans_" + key.lower() + ".JPG",
                cv2.cvtColor(transformations[key], cv2.COLOR_BGR2RGB),
            )
        else:
            ax = fig.add_subplot(gs[0, i])
            ax.imshow(transformations[key])
            ax.axis("off")
            ax.set_title(key)
    ax = fig.add_subplot(gs[1, :])
    if save:
        color_hist_pcv(img, save=True, src=path, dest=dest)
        plt.close()
        return
    color_hist_pcv(img, ax)
    plt.show()
    plt.close()


def blur_pcv(gray_img: MatLike) -> MatLike:
    bin_img = pcv.threshold.otsu(gray_img=gray_img, object_type="dark")
    blur = pcv.gaussian_blur(img=bin_img, ksize=(9, 9), sigma_x=0, sigma_y=None)
    return blur


def mask_pcv(img: MatLike, background: MatLike) -> MatLike:
    gray_img = pcv.rgb2gray_hsv(rgb_img=img, channel="s")

    mask = pcv.threshold.otsu(gray_img=gray_img, object_type="dark")
    mask_back = pcv.threshold.otsu(gray_img=background, object_type="dark")

    masked_image = pcv.apply_mask(img=img, mask=mask, mask_color="white")
    masked_image = pcv.apply_mask(img=img, mask=mask_back, mask_color="white")

    return masked_image


def plot_image(img: MatLike, filter: str) -> None:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    plt.axis("off")
    plt.title(filter.capitalize())
    plt.show()


def apply_filter(
    path: pathlib.Path,
    dest: pathlib.Path,
    filter: str,
    save: bool = False,
) -> None:
    img = cv2.imread(path)
    gray_img = pcv.rgb2gray_lab(rgb_img=img, channel="a")
    match filter:
        case "intensity":
            color_hist_pcv(img, save=save, src=path, dest=dest)
            return
        case "all":
            handle_all(img, path=path, dest=dest, save=save, gray_img=gray_img)
            return

        case "blur":
            filtered = blur_pcv(gray_img)
        case "mask":
            filtered = mask_pcv(img, gray_img)
        case "roi":
            filtered = roi_pcv(img, gray_img)
        case "analyze":
            filtered = analyze_pcv(img, gray_img)
        case "landmarks":
            filtered = landmarks_pcv(img, gray_img)

    if not save:
        plot_image(filtered, filter)
        return

    tmp = str(dest) + "/" + path.name[:-4] + "_trans_" + filter + ".JPG"
    cv2.imwrite(tmp, filtered)


def handle_dir(
    args: argparse.Namespace,
    src_path: pathlib.Path,
    dest_path: pathlib.Path,
):
    for root, _, filenames in os.walk(src_path):
        images = [root + "/" + image for image in filenames if image.endswith(".JPG")]
        if len(images) == 0:
            continue
        tmp_dst = pathlib.Path(root.replace(str(src_path), str(dest_path)))
        if not pathlib.Path.exists(tmp_dst):
            tmp_dst.mkdir(parents=True)
        for file in filenames:
            apply_filter(
                pathlib.Path(root + "/" + file), tmp_dst, args.transformation, save=True
            )


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

    src_path = pathlib.Path(args.src)

    if not pathlib.Path.exists(src_path):
        sys.exit("Source path does not exists")

    dest_path = pathlib.Path(args.destination if args.destination else args.src)

    if not pathlib.Path.exists(dest_path):
        dest_path.mkdir()

    if pathlib.Path.is_dir(src_path):
        handle_dir(args, src_path, dest_path)
    else:
        apply_filter(src_path, dest_path, args.transformation)


if __name__ == "__main__":
    main()
