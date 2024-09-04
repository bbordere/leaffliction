from cv2.typing import MatLike
import numpy as np
import cv2


def flip(img: MatLike) -> MatLike:
    """Flip image 90 degrees vertically

    Args:
        img (MatLike): img to flip

    Returns:
        MatLike: flipped img
    """
    flipped = cv2.flip(img, 1)
    return flipped


def rotate(img: MatLike, angle: int = 30) -> MatLike:
    """Rotate image counter clock wise

    Args:
        img (MatLike): img to rotate
        angle (int, optional): rotation angle target. Defaults to 30.

    Returns:
        MatLike: rotated img
    """
    (h, w) = img.shape[:2]

    center = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    rotated_image = cv2.warpAffine(img, M, (new_w, new_h))
    rotated_image = cv2.resize(rotated_image, (256, 256))

    return rotated_image


def blur(img: MatLike, ksize: tuple = (5, 5)) -> MatLike:
    """Apply gaussian blur to image

    Args:
        img (MatLike): image to be blurred
        ksize (tuple[int, int], optional): kernel size for blur effect. \
            Defaults to (7, 7).

    Returns:
        MatLike: blurred img
    """
    blurred = cv2.GaussianBlur(img, ksize, 0)
    return blurred


def contrast(img: MatLike, alpha: float = 1.3, beta: float = 1) -> MatLike:
    """Apply filter to change contrast

    Args:
        img (MatLike): image to be altered
        alpha (float, optional): contrast factor. Defaults to 1.3.
        beta (float, optional): brigthness factor. Defaults to 1.

    Returns:
        MatLike: altered image
    """
    contrasted = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return contrasted


def scaling(img: MatLike, factor: float = 1.35) -> MatLike:
    """Rescale image around center
    Args:
        img (MatLike): image to rescale
        factor (float, optional): factor rescale. Defaults to 1.35.

    Returns:
        MatLike: rescaled image
    """
    (h, w) = img.shape[:2]

    new_w = int(w / factor)
    new_h = int(h / factor)

    start_x = (w - new_w) // 2
    start_y = (h - new_h) // 2

    cropped_image = img[start_y : start_y + new_h, start_x : start_x + new_w]

    zoomed_image = cv2.resize(
        cropped_image, (w, h), interpolation=cv2.INTER_LINEAR
    )
    return zoomed_image


def project(img: MatLike) -> MatLike:
    """Project image

    Args:
        img (MatLike): image to project

    Returns:
        _type_: projected image
    """
    src_pts = np.float32([[0, 0], [255, 0], [0, 255], [255, 255]])
    dst_pts = np.float32([[50, 50], [200, 30], [30, 200], [220, 220]])

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # warped_image = cv2.warpPerspective(
    #     img,
    #     M,
    #     (256, 256),
    #     borderValue=(0, 0, 0),
    #     flags=cv2.INTER_CUBIC,
    #     borderMode=cv2.BORDER_REPLICATE,
    # )

    warped_image = cv2.warpPerspective(img, M, (256, 256))
    return warped_image


# filter look-up table for applying augmentation
filters_table = {
    "Flip": flip,
    "Rotation": rotate,
    "Blur": blur,
    "Contrast": contrast,
    "Scaling": scaling,
    "Projective": project,
}

extensions = ["_" + key for key in filters_table]

params = {
    "Blue": "blue",
    "Blue-Yellow": "yellow",
    "Green": "green",
    "Green-Magenta": "magenta",
    "Hue": "purple",
    "Lightness": "gray",
    "Red": "red",
    "Saturation": "cyan",
    "Value": "orange",
}
