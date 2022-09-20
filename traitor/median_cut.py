import numpy as np
import skimage
from typing import Optional


def _median_cut(
    image: np.ndarray,
    depth: int,
    mask: np.ndarray,
    idx: int = 1,
) -> np.ndarray:
    if depth == 0:
        return mask

    # get channel with highest "range" of values, in this case std
    # TODO: could also use range, i.e. max-min here
    c_idx = np.argmax(image[mask == idx].std(axis=0))

    # get median of channel with highest range
    median = np.median(image[mask == idx, c_idx])

    # split by median
    mask_l = (image[:, :, c_idx] < median) & (mask == idx)
    mask_h = (image[:, :, c_idx] >= median) & (mask == idx)

    idx_l = idx << 1
    idx_h = (idx << 1) + 1
    mask[mask_l] = idx_l
    mask[mask_h] = idx_h

    # recurse
    _median_cut(image=image, mask=mask, depth=depth - 1, idx=idx_l)
    _median_cut(image=image, mask=mask, depth=depth - 1, idx=idx_h)

    return mask


def median_cut(
    image: np.ndarray,
    depth: int,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    if mask is None:
        mask = np.full((image.shape[0], image.shape[1]), fill_value=1, dtype=int)
    mask = mask.astype(int)
    mask[mask != 0] = 1

    cluster_membership = _median_cut(image, depth, mask.copy(), idx=1)

    # recode: 0 is background, 1,2,...,n are color cluster memberships
    for i, cl in enumerate(np.unique(cluster_membership)):
        cluster_membership[cluster_membership == cl] = i

    return cluster_membership


def dominant_colors_mc(image: np.ndarray, cluster_membership: np.ndarray):
    colors = []
    counts = []
    for cl in np.unique(cluster_membership):
        if cl == 0:
            continue
        idcs = cluster_membership == cl
        colors.append(image[idcs].mean(0))
        counts.append(np.sum(idcs))
    return np.array(colors), np.array(counts)


def dominant_color_image_mc(image: np.ndarray, cluster_membership: np.ndarray):
    new_image = np.zeros_like(image)
    for cl in np.unique(cluster_membership):
        if cl == 0:
            continue
        new_image[cluster_membership == cl] = image[cluster_membership == cl].mean(0)
    return new_image
