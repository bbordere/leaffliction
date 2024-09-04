import argparse
from plantcv import plantcv as pcv
import cv2
import pathlib
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Optional
import multiprocessing
from tensorflow.keras.preprocessing import image
from tqdm import tqdm


class ImageTransformer:
    """Image Transformer class"""

    def __init__(self) -> None:
        self.img = None
        self.img_no_back = None
        self.back_mask = None
        self.dis_mask = None
        self.contours_mask = None
        self.f_table = {
            "blur": self.gaussian_blur,
            "mask": self.mask,
            "roi": self.roi,
            "analyze": self.analyze,
            "landmarks": self.landmarks,
            "canny": self.canny_edges,
        }

    def open(
        self, path: pathlib.Path, size: tuple = (256, 256), pred: bool = False
    ):
        """open image specified on path

        Args:
            path (pathlib.Path): path to img
            size (tuple, optional): target size for image. Defaults to (256, 256).
            pred (bool, optional): prediction flag. Defaults to False.
        """

        self.original_img = cv2.imread(str(path))
        self.img = image.load_img(path, target_size=size)
        self.img = image.img_to_array(self.img)
        self.img = self.img.astype("uint8")

        if not pred:
            self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

        self.back_mask = self.compute_back_mask()
        self.dis_mask = self.compute_dis_mask()

    def compute_dis_mask(self) -> np.ndarray:
        """compute disease mask for applying filters

        Returns:
            np.ndarray: disease binary mask
        """
        gray_img = pcv.rgb2gray_lab(rgb_img=self.img_no_back, channel="a")
        thresh = pcv.threshold.triangle(gray_img=gray_img, object_type="dark")
        return thresh

    def compute_back_mask(self) -> np.ndarray:
        """compute background binary mask

        Returns:
            np.ndarray: _description_
        """
        balanced_img = pcv.white_balance(self.img, mode="hist")
        gray_img = pcv.rgb2gray_lab(rgb_img=balanced_img, channel="b")
        thresh = pcv.threshold.triangle(gray_img=gray_img, object_type="light")
        thresh = pcv.fill(bin_img=thresh, size=200)
        mask = pcv.fill(bin_img=pcv.invert(gray_img=thresh), size=200)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours_mask = np.ones_like(mask)
        res = self.contours_mask.copy()
        if len(contours):
            cv2.drawContours(
                self.contours_mask,
                contours[np.argmax([len(c) for c in contours])],
                -1,
                (0, 0, 0),
                -1,
            )
            cv2.fillPoly(res, pts=contours, color=(0, 0, 0))

        self.img_no_back = pcv.apply_mask(self.img, res, "white")
        try:
            res = pcv.fill_holes(res)
        except RuntimeError:
            pass
        return res

    def gaussian_blur(self, ksize: tuple = (3, 3)) -> np.ndarray:
        """apply gaussian blur on image

        Args:
            ksize (tuple, optional): kernel size for blur. Defaults to (3, 3).

        Returns:
            np.ndarray: blured image
        """
        return pcv.gaussian_blur(self.dis_mask, ksize)

    def analyze(self) -> np.ndarray:
        """apply analyze filter on image

        Returns:
            np.ndarray: filtered image
        """
        roi = pcv.roi.rectangle(
            img=self.back_mask,
            x=0,
            y=0,
            w=self.dis_mask.shape[0],
            h=self.dis_mask.shape[1],
        )
        roi_mask = pcv.roi.filter(
            mask=self.dis_mask, roi=roi, roi_type="partial"
        )
        return pcv.analyze.size(img=self.img, labeled_mask=roi_mask)

    def roi(self) -> np.ndarray:
        """apply roi filter on image

        Returns:
            np.ndarray: filtered image
        """
        mask = pcv.invert(self.dis_mask)
        mask = pcv.invert(pcv.logical_xor(self.back_mask, mask))

        roi = pcv.roi.rectangle(
            img=mask,
            x=0,
            y=0,
            w=self.dis_mask.shape[0],
            h=self.dis_mask.shape[1],
        )
        roi_mask = pcv.roi.filter(mask=mask, roi=roi, roi_type="partial")
        roi_mask = pcv.logical_and(roi_mask, self.contours_mask)

        color_pix = np.where(mask != 0)

        img_roi = self.img.copy()
        try:
            x, y, w, h = (
                np.min(color_pix[1]),
                np.min(color_pix[0]),
                np.max(color_pix[1]) - np.min(color_pix[1]),
                np.max(color_pix[0]) - np.min(color_pix[0]),
            )
            img_roi[roi_mask != 0] = (0, 255, 0)
            cv2.rectangle(img_roi, (x, y), (x + w, y + h), (255, 0, 0), 2)
        except ValueError:
            cv2.rectangle(
                img_roi,
                (0, 0),
                (self.dis_mask.shape[0], self.dis_mask.shape[1]),
                (255, 0, 0),
                2,
            )
        return img_roi

    def landmarks(self) -> np.ndarray:
        """apply pseudo landmarks filter on image

        Returns:
            np.ndarray: filtered image
        """
        res = self.img.copy()
        left, right, center = pcv.homology.x_axis_pseudolandmarks(
            img=self.img, mask=self.back_mask
        )

        for pl, pc, pr in zip(
            left.astype(int), right.astype(int), center.astype(int)
        ):
            cv2.circle(
                res, pl[0], radius=5, color=(209, 23, 206), thickness=-1
            )
            cv2.circle(res, pr[0], radius=5, color=(49, 14, 204), thickness=-1)
            cv2.circle(res, pc[0], radius=5, color=(222, 86, 18), thickness=-1)
        return res

    def mask(self) -> np.ndarray:
        """apply mask filter on image

        Returns:
            np.ndarray: filtered image
        """
        return pcv.apply_mask(self.img_no_back, self.dis_mask, "white")

    def canny_edges(self):
        """apply canny edges filter on image

        Returns:
            np.ndarray: filtered image
        """
        res = self.img.copy()
        canny_edges_contours = pcv.canny_edge_detect(self.img_no_back)
        contours, _ = cv2.findContours(
            canny_edges_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        cv2.drawContours(res, contours, -1, (0, 255, 0), 1)
        return pcv.canny_edge_detect(self.img_no_back, high_thresh=145)

    def all_filter(
        self, src: pathlib.Path, dest: pathlib.Path, save: bool = False
    ):
        """handle all filters option

        Args:
            src (pathlib.Path): path to src image
            dest (pathlib.Path): path to destination
            save (bool, optional): save to disk. Defaults to False.
        """
        transformations = {
            "Original": cv2.cvtColor(self.img, cv2.COLOR_RGB2BGR),
            "Blur": cv2.cvtColor(self.gaussian_blur(), cv2.COLOR_RGB2BGR),
            "Mask": cv2.cvtColor(self.mask(), cv2.COLOR_RGB2BGR),
            "ROI": cv2.cvtColor(self.roi(), cv2.COLOR_RGB2BGR),
            "Analyze": cv2.cvtColor(self.analyze(), cv2.COLOR_RGB2BGR),
            "Landmarks": cv2.cvtColor(self.landmarks(), cv2.COLOR_RGB2BGR),
            "Canny": cv2.cvtColor(self.canny_edges(), cv2.COLOR_RGB2BGR),
        }
        fig, ax = plt.subplots(1, 7)
        for i, (key, transformed_img) in enumerate(transformations.items()):
            if save:
                output_path = dest / f"{src.stem}_trans_{key.lower()}.JPG"
                cv2.imwrite(
                    str(output_path),
                    cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB),
                )
            else:
                ax[i].imshow(transformed_img)
                ax[i].axis("off")
                ax[i].set_title(key)
        if not save:
            plt.show()
        plt.close(fig)

    def star_filter(self, args: list):
        """handler function for multiprocessing

        Args:
            args (list): args for callbacks

        Returns:
            any: routine returns value
        """
        return self.filter(*args)

    def filter(
        self,
        src: pathlib.Path,
        transformation: str,
        save: bool = False,
        dest: Optional[pathlib.Path] = None,
    ):
        """apply filter on image

        Args:
            src (pathlib.Path): path to image
            transformation (str): transformation to apply
            save (bool, optional): save to disk. Defaults to False.
            dest (Optional[pathlib.Path], optional): dest path if save. Defaults to None.

        """
        self.open(src)
        if transformation == "all":
            self.all_filter(src, dest, save)
            return
        filtered = self.f_table[transformation]()
        if not save:
            plt.imshow(cv2.cvtColor(filtered, cv2.COLOR_RGB2BGR))
            plt.axis("off")
            plt.title(transformation.capitalize())
            plt.show()
            plt.close()
            return
        tmp = (
            str(dest)
            + "/"
            + src.name[:-4]
            + "_trans_"
            + transformation
            + ".JPG"
        )
        cv2.imwrite(tmp, filtered)
        return True

    def handle_dir(
        self,
        src: pathlib.Path,
        transformation: str,
        dest: Optional[pathlib.Path] = None,
    ):
        """handling function for directory

        Args:
            src (pathlib.Path): path to directory
            transformation (str): transformation to apply
            dest (Optional[pathlib.Path], optional): dest path if save. Defaults to None.
        """
        for root, _, filenames in os.walk(src):
            images = [
                root + "/" + image
                for image in filenames
                if image.endswith(".JPG")
            ]
            if len(images) == 0:
                continue
            tmp_dst = pathlib.Path(root.replace(str(src), str(dest)))
            if not pathlib.Path.exists(tmp_dst):
                tmp_dst.mkdir(parents=True)

            fct_args = list(
                (
                    pathlib.Path(root + "/" + file),
                    transformation,
                    True,
                    tmp_dst,
                )
                for file in filenames
            )

            with multiprocessing.Pool() as pool:
                _ = list(
                    tqdm(
                        pool.imap(self.star_filter, fct_args),
                        total=len(fct_args),
                        desc=root,
                    )
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="Transformation",
        description="Apply transformation on image or folder",
    )
    parser.add_argument("src", help="path to image or directory")
    parser.add_argument(
        "-d", "--destination", help="path to image or directory"
    )
    parser.add_argument(
        "-t",
        "--transformation",
        help="trans",
        default="all",
        choices=[
            "all",
            "blur",
            "mask",
            "roi",
            "analyze",
            "landmarks",
            "canny",
        ],
    )

    args = parser.parse_args()

    src_path = pathlib.Path(args.src)

    if not pathlib.Path.exists(src_path):
        sys.exit("Source path does not exists")

    dest_path = pathlib.Path(
        args.destination if args.destination else args.src
    )

    if not pathlib.Path.exists(dest_path):
        dest_path.mkdir()

    transformer = ImageTransformer()

    if pathlib.Path.is_dir(src_path):
        transformer.handle_dir(src_path, args.transformation, dest_path)
    else:
        transformer.filter(src_path, args.transformation)


if __name__ == "__main__":
    main()
