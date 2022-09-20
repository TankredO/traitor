import warnings
from pathlib import Path

import skimage
import skimage.io
import skimage.measure
import skimage.morphology
import skimage.draw
import skimage.transform
import skimage.color
import sklearn.cluster
import cv2
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import cdist, euclidean

from typing import Iterable, Tuple, Optional, List


def segment_image(
    image: np.ndarray,
    k: int = 3,
    bg_col: Optional[Iterable[int]] = None,
    n_bg_clusters: int = 1,
) -> np.ndarray:
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    _, labels, (centers) = cv2.kmeans(
        pixel_values,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    labels: np.ndarray = labels.flatten()  # type: ignore

    # assume that the cluster with highest number of members (pixels) is background
    if bg_col is None:
        lab, counts = np.unique(labels, return_counts=True)
        bg_labels = np.argsort(counts)[::-1][:n_bg_clusters]
    # find cluster closest to background color
    else:
        bg_col = np.array(bg_col)
        centers = np.uint8(centers)
        distances = np.sqrt(np.sum((bg_col - centers) ** 2, axis=1))

        bg_labels = np.argsort(distances)[:n_bg_clusters]

    bin_image = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    bin_image[(~np.isin(labels, bg_labels)).reshape(bin_image.shape)] = True

    return bin_image


def segment_image2(
    image: np.ndarray,
    k: int = 3,
    bg_col: Optional[Iterable[int]] = None,
    n_bg_clusters: int = 1,
) -> np.ndarray:
    # lab color space to ignore "brightness"
    image = skimage.color.rgb2lab(image)

    pixel_values = image[:, :, [1, 2]].reshape((-1, 2)).astype(np.float32)

    _, labels, (centers) = cv2.kmeans(
        pixel_values,
        k,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.1),
        10,
        cv2.KMEANS_RANDOM_CENTERS,
    )
    labels: np.ndarray = labels.flatten()  # type: ignore

    # assume that the cluster with highest number of members (pixels) is background
    if bg_col is None:
        lab, counts = np.unique(labels, return_counts=True)
        bg_labels = np.argsort(counts)[::-1][:n_bg_clusters]
    # find cluster closest to background color
    else:
        bg_col = skimage.color.rgb2lab(np.array(bg_col))[[1, 2]]
        centers = np.uint8(centers)
        distances = np.sqrt(np.sum((bg_col - centers) ** 2, axis=1))

        bg_labels = np.argsort(distances)[:n_bg_clusters]

    bin_image = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    bin_image[(~np.isin(labels, bg_labels)).reshape(bin_image.shape)] = True

    return bin_image


def _segment_image_approx(
    image: np.ndarray,
    k: int = 2,
    bg_col: Optional[Iterable[int]] = None,
    n_bg_clusters: int = 1,
    n_pixels: int = 10000,
) -> np.ndarray:
    pixel_values = image.reshape((-1, 3)).astype(np.float32)

    import sklearn.cluster

    km = sklearn.cluster.KMeans(n_clusters=k)
    km.fit(
        pixel_values[np.random.choice(np.arange(pixel_values.shape[0]), n_pixels), :]
    )

    labels = km.predict(pixel_values)
    centers = km.cluster_centers_

    # assume that cluster with most pixels is background
    if bg_col is None:
        _, counts = np.unique(labels, return_counts=True)
        bg_labels = np.argsort(counts)[::-1][:n_bg_clusters]
    # find cluster closest to background color
    else:
        bg_col = np.array(bg_col)
        centers = np.uint8(centers)
        distances = np.sqrt(np.sum((bg_col - centers) ** 2, axis=1))

        bg_labels = np.argsort(distances)[:n_bg_clusters]

    bin_image = np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    bin_image[(~np.isin(labels, bg_labels)).reshape(bin_image.shape)] = True

    return bin_image


def get_contours(bin_image: np.ndarray, min_size: float) -> List[np.ndarray]:
    bin_image = bin_image.copy()
    bin_image[:, 0] = False
    bin_image[:, -1] = False
    bin_image[0, :] = False
    bin_image[-1, :] = False

    contours, hierarchy = cv2.findContours(
        bin_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    contours = [c[:, -1, [1, 0]] for c in contours]
    contours = [c for c in contours if polygon_area(c[:, [1, 0]]) > min_size]

    return contours


def _count_objects_approx(
    bin_image: np.ndarray,
    scale: Optional[float] = None,
) -> int:
    if scale is None:
        scale = 1000 / np.max(bin_image.shape)

    contours, _ = cv2.findContours(
        skimage.transform.rescale(bin_image, scale).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    contours = [c[:, -1, [1, 0]] for c in contours]

    return len(contours)


def segment_image_adaptive(
    image: np.ndarray,
    bg_col: Optional[Iterable[int]] = None,
    n_pixels: int = 10000,
    scale: Optional[float] = None,
) -> np.ndarray:
    bin_image_k2 = _segment_image_approx(
        image, k=2, bg_col=bg_col, n_bg_clusters=1, n_pixels=n_pixels
    )
    bin_image_k3 = _segment_image_approx(
        image, k=3, bg_col=bg_col, n_bg_clusters=1, n_pixels=n_pixels
    )

    counts_k2 = _count_objects_approx(bin_image_k2, scale=scale)
    counts_k3 = _count_objects_approx(bin_image_k3, scale=scale)

    diff_counts = abs(counts_k2 - counts_k3)
    max_count = max(counts_k2, counts_k3)
    if diff_counts / max_count < 0.20:
        warnings.warn(
            "Adaptive segmentation uncertain. Please check the segmentation results."
        )

    if counts_k2 <= counts_k3:
        return bin_image_k2

    return bin_image_k3


def polygon_area(xy: np.ndarray) -> np.ndarray:
    return 0.5 * np.abs(
        np.dot(xy[:, 0], np.roll(xy[:, 1], 1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], 1))
    )


def resample_polygon(xy: np.ndarray, n_points: int = 100) -> np.ndarray:
    # Cumulative Euclidean distance between successive polygon points.
    # This will be the "x" for interpolation
    d = np.cumsum(np.r_[0, np.sqrt((np.diff(xy, axis=0) ** 2).sum(axis=1))])

    # get linearly spaced points along the cumulative Euclidean distance
    d_sampled = np.linspace(0, d.max(), n_points)

    # interpolate x and y coordinates
    xy_interp = np.c_[
        np.interp(d_sampled, d, xy[:, 0]),
        np.interp(d_sampled, d, xy[:, 1]),
    ]

    return xy_interp


def get_minsize_adaptive(
    bin_image: np.ndarray,
) -> int:
    import sklearn
    import sklearn.cluster

    contours = get_contours(bin_image, min_size=1)
    contours_sizes = np.array([polygon_area(c) for c in contours])

    _, cl, _ = sklearn.cluster.k_means(contours_sizes.reshape(-1, 1), 2)
    cl_mean_sizes = [np.mean(contours_sizes[cl == i]) for i in np.unique(cl)]
    max_noise_size = contours_sizes[cl == np.argmin(cl_mean_sizes)].max()
    min_obj_size = contours_sizes[cl == np.argmax(cl_mean_sizes)].min()

    min_size = np.ceil(max_noise_size)

    if min_size > min_obj_size:
        warnings.warn(
            "Could not clearly separate noise from objects. Please check segmentation results."
        )

    return min_size


def get_minsize_adaptive2(
    bin_image: np.ndarray,
) -> int:
    import sklearn
    import sklearn.cluster

    contours = get_contours(bin_image, min_size=1)
    contours_sizes = np.array([polygon_area(c) for c in contours])

    min_size = contours_sizes.mean() * 1.0 / 3.0

    return min_size


def filter_bin_image(
    bin_image: np.ndarray,
    min_size: int = 256,
    area_threshold: int = 256,
) -> np.ndarray:
    bin_image = skimage.morphology.binary_erosion(bin_image)
    bin_image = skimage.morphology.binary_dilation(bin_image)
    bin_image = skimage.morphology.remove_small_objects(bin_image, min_size=min_size)
    bin_image = skimage.morphology.remove_small_holes(
        bin_image, area_threshold=area_threshold
    )

    return bin_image


def get_bbox(
    contour: np.ndarray,
    padding: Iterable[int] = (5, 5, 5, 5),
    image_shape: Optional[Iterable[int]] = None,
) -> np.ndarray:
    padding = np.array(padding)

    y1, x1 = np.floor(contour.min(axis=0)).astype(int) - [padding[0], padding[3]]
    y2, x2 = np.ceil(contour.max(axis=0)).astype(int) + [padding[1], padding[2]]

    y1 = max(0, y1)
    x1 = max(0, x1)
    y2 = max(0, y2)
    x2 = max(0, x2)
    if not image_shape is None:
        image_shape = np.array(image_shape)
        y1 = min(y1, image_shape[0] - 1)
        x1 = min(x1, image_shape[1] - 1)
        y2 = min(y2, image_shape[0] - 1)
        x2 = min(x2, image_shape[1] - 1)

    return np.array([x1, y1, x2, y2])


def extract_subimage(
    contour: np.ndarray,
    image: np.ndarray,
    padding: Iterable[int] = (5, 5, 5, 5),
    remove_background: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    padding = np.array(padding)

    bbox = get_bbox(contour=contour, padding=padding, image_shape=image.shape)

    sub_image: np.ndarray = image[bbox[1] : bbox[3], bbox[0] : bbox[2]].copy()
    mask = skimage.draw.polygon2mask(
        (sub_image.shape[0], sub_image.shape[1]), contour - [bbox[1], bbox[0]]
    )
    if remove_background:
        sub_image[~mask, :] = 0

    return sub_image, mask, bbox


def rotate_upright(points: np.ndarray, box_type: str = "ellipse"):
    if box_type == "box":
        xy, wh, angle = cv2.minAreaRect(points)
    elif box_type == "ellipse":
        xy, wh, angle = cv2.fitEllipse(points)
    else:
        msg = f'Unknown box_type "{box_type}".'
        raise Exception(msg)

    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))

    points_rot: np.ndarray = points - points.mean(axis=0)
    points_rot = points_rot.dot(R)
    points_rot += points.mean(axis=0)

    return points_rot, angle


def align_shapes(a: np.ndarray, b: np.ndarray):
    a = np.array(a, dtype=np.double, copy=True)
    b = np.array(b, dtype=np.double, copy=True)

    a -= np.mean(a, 0)
    b -= np.mean(b, 0)

    norm1 = np.linalg.norm(a)
    norm2 = np.linalg.norm(b)

    a /= norm1
    b /= norm2

    best_rotation, best_scale = orthogonal_procrustes(a, b)
    best_disparity = np.sum(np.square(a - b.dot(best_rotation.T)))
    best_i = 0
    for i in range(1, b.shape[0]):
        cur_b = np.r_[
            b[
                i:,
            ],
            b[
                :i,
            ],
        ]

        rot, scale = orthogonal_procrustes(a, cur_b)
        disparity = np.sum(np.square(a - cur_b.dot(rot.T)))

        if disparity < best_disparity:
            best_disparity = disparity
            best_rotation = rot
            best_scale = scale
            best_i = i

    return b.dot(best_rotation.T), best_i, best_rotation, best_scale, best_disparity


def primary_colors(
    image: np.ndarray, mask: Optional[np.ndarray] = None, n_colors: int = 5
):
    km = sklearn.cluster.KMeans(n_clusters=n_colors, n_init=3)

    if not mask is None:
        pixel_values = image[mask != 0].reshape((-1, image.shape[2]))
    else:
        pixel_values = image.reshape((-1, image.shape[2]))

    cl: np.ndarray = km.fit_predict(pixel_values)
    _, counts = np.unique(cl, return_counts=True)
    colors: np.ndarray = km.cluster_centers_
    colors = colors.clip(min=0.0)

    order = np.flip(np.argsort(counts))

    colors = colors[order]
    counts = counts[order].astype(int)

    return colors, counts


def resize_image(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    order: int = 1,
):
    h, w = image.shape[:2]

    if width is None and height is None:
        return image

    if not height is None:
        ratio = height / h
        width = int(w * ratio)
    elif not width is None:
        ratio = width / w
        height = int(h * ratio)

    return skimage.transform.resize(image, (height, width), order=order)


def reset_starting_point(contour: np.ndarray) -> np.ndarray:
    best_i = 0
    best_d = np.inf
    mid_x = (contour[:, 1].max() - contour[:, 1].min()) / 2 + contour[:, 1].min()
    mid_y = (contour[:, 0].max() - contour[:, 0].min()) / 2 + contour[:, 0].min()
    for i in range(contour.shape[0]):
        d = np.abs(contour[i, 1] - mid_x)
        if d < best_d and contour[i, 0] < mid_y:
            best_d = d
            best_i = i
    return np.r_[
        contour[
            best_i:,
        ],
        contour[
            :best_i,
        ],
    ]


# source: https://stackoverflow.com/a/30305181/5665958
def geometric_median(X, eps=1e-5):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros / r
            y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

        if euclidean(y, y1) < eps:
            return y1

        y = y1
