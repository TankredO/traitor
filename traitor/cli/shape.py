import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from cloup import (
    group,
    command,
    option,
    help_option,
    Path as PathType,
    Choice,
    option_group,
    IntRange,
)


@command(
    "shape",
    help="Calculate shape clusters.",
    no_args_is_help=True,
)
@option_group(
    "Required options",
    option(
        "-i",
        "--measurements_file",
        type=PathType(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help="Input measurements (CSV) file as produced by the measure command.",
        required=True,
    ),
    option(
        "-o",
        "--out_dir",
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f"""
        Output directory. Will be created if it does not exist. By default
        a directory with the name "<MEASUREMENTS_FILE_NAME>_shape"
        will be created in the current working directory ({Path(".").resolve()}).
    """,
        default=None,
        required=False,
    ),
)
@option_group(
    "Clustering options",
    option(
        "-k",
        "--n_clusters",
        type=IntRange(min=2, max_open=True),
        help="Number of shape clusters.",
        default=5,
        show_default=True,
    ),
    option(
        "-d",
        "--descriptors",
        type=IntRange(min=1, max_open=True),
        help="Number of Fourier descriptors to use for shape approximation.",
        default=10,
        show_default=True,
    ),
    option(
        "-p",
        "--n_pcs",
        type=IntRange(min=1, max_open=True),
        help="Number of principal coordinates of the Fourier descriptors to retain for clustering.",
        default=5,
        show_default=True,
    ),
)
@help_option("-h", "--help")
def shape(
    measurements_file: Path,
    out_dir: Optional[Path],
    n_clusters: int,
    descriptors: int,
    n_pcs: int,
):
    if out_dir is None:
        measurements_file_name = measurements_file.with_suffix("").name
        out_dir = Path(f"{measurements_file_name}_shape")

    try:
        out_dir.mkdir(parents=True, exist_ok=True)  # !!!!!
    except FileExistsError as err:
        msg = f'ERROR: output directory "{out_dir}" already exists.\n'
        sys.stderr.write(msg)
        sys.exit(1)

    import pandas as pd
    import numpy as np
    import sklearn.decomposition
    import sklearn.cluster
    import pyefd
    import sys
    import tqdm
    from ..defaults import DEFAULT_N_CONTOUR_VERTICES

    measurements = pd.read_csv(measurements_file)

    # reshape [n, vertices, xy]
    contours = measurements.iloc[:, -DEFAULT_N_CONTOUR_VERTICES * 2 :].values.reshape(
        -1, DEFAULT_N_CONTOUR_VERTICES, 2
    )

    # check if NaN in contours. NaN values indicate that the extraction was not successful
    nan_row_idcs = np.array([np.any(np.isnan(c)) for c in contours])
    if sum(nan_row_idcs) > 0:
        nan_measurements: pd.DataFrame = measurements.iloc[nan_row_idcs]
        nan_groups = nan_measurements["group"].unique()
        msg = f"WARNING: Encountered NaN. Please check extractions for the following groups:\n"
        msg = (
            msg
            + "\n".join(nan_groups)
            + "\n"
            + "Ignoring contours containing NaN values ...\n"
        )
        sys.stderr.write(msg)
    measurements = measurements[~nan_row_idcs]
    contours = contours[~nan_row_idcs]

    # elliptic Fourier analysis
    coeffs = []
    print("Calculating Fourier descriptors ...")
    for c in tqdm.tqdm(contours):
        try:
            coeffs.append(pyefd.elliptic_fourier_descriptors(c, order=descriptors))
        except RuntimeWarning as w:
            print(w)
    print("Done.")

    # PCA
    print("Running PCA ...")
    x = np.array([c.ravel() for c in tqdm.tqdm(coeffs)])
    pca = sklearn.decomposition.PCA(n_pcs)
    pca.fit(x)
    r = pca.transform(x)
    print("Done.")

    # K-Means
    km = sklearn.cluster.KMeans(n_clusters=n_clusters)
    print("Fitting K-Means ...")
    km.fit(r)
    print("Done.")
    cl = km.predict(r)

    # write outputs
    import pickle

    pca_file = out_dir / "pca.pkl"
    with open(pca_file, "wb") as f:
        pickle.dump(pca, f)

    km_file = out_dir / "km.pkl"
    with open(km_file, "wb") as f:
        pickle.dump(km, f)

    coeff_df = pd.DataFrame(
        x,
        columns=[
            f"{d}_{i}"
            for d, i in zip(
                ["A", "B", "C", "D"] * x.shape[1], np.repeat(np.arange(descriptors), 4)
            )
        ],
        index=measurements.index,
    )
    coeff_df = pd.concat(
        (measurements[["image_name", "image_file", "mask_file", "group"]], coeff_df),
        axis=1,
    )
    coeff_file = out_dir / "coeffs.csv"
    coeff_df.to_csv(coeff_file, index=False)

    pc_df = pd.DataFrame(
        r, columns=[f"PC_{i}" for i in np.arange(n_pcs)], index=measurements.index
    )
    pc_df = pd.concat(
        (measurements[["image_name", "image_file", "mask_file", "group"]], pc_df),
        axis=1,
    )
    pc_file = out_dir / "pcs.csv"
    pc_df.to_csv(pc_file, index=False)

    cl_df = pd.DataFrame(cl, columns=["cluster"], index=measurements.index)
    cl_df = pd.concat(
        (measurements[["image_name", "image_file", "mask_file", "group"]], cl_df),
        axis=1,
    )
    cl_file = out_dir / "cl.csv"
    cl_df.to_csv(cl_file, index=False)

    # visualize
    import matplotlib.pyplot as plt
    from ..tools import geometric_median

    fig, axs = plt.subplots(1, n_clusters, figsize=[30, 7])
    for i in range(n_clusters):
        r_tmp = r[cl == i, :]
        med = geometric_median(r_tmp)
        m = pyefd.reconstruct_contour(
            pca.inverse_transform(med).reshape(-1, 4), num_points=50
        )
        axs[i].plot(m[:, 0], m[:, 1], color="green", zorder=10, linewidth=3)

        n = 25
        if r_tmp.shape[0] < n:
            r_idcs = np.arange(r_tmp.shape[0])
        else:
            r_idcs = np.random.choice(np.arange(r_tmp.shape[0]), size=n)
        for c in r_tmp[r_idcs, :]:
            yx = pyefd.reconstruct_contour(pca.inverse_transform(c).reshape(-1, 4))
            axs[i].plot(yx[:, 0], yx[:, 1], alpha=0.25, color="orange")

        axs[i].set_aspect("equal")
        axs[i].set_title(f"Cluster {i}")
    plt.show()

    import matplotlib.patches as patches

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    cols = plt.cm.tab20(np.unique(cl))
    for cl_idx in np.unique(cl):
        pcs_subset = r[cl == cl_idx]
        ax.scatter(
            pcs_subset[:, 0],
            pcs_subset[:, 1],
            c=cols[cl_idx].reshape(1, 4),
            alpha=0.25,
        )

        med = geometric_median(pcs_subset)
        med_contour = pyefd.reconstruct_contour(
            pca.inverse_transform(med).reshape(-1, 4)
        )
        med_poly = patches.Polygon(
            med[:2] + med_contour[:, :2] * 0.1,
            edgecolor="black",
            facecolor=cols[cl_idx],
            zorder=10,
            alpha=0.5,
        )
        ax.add_patch(med_poly)
    plt.show()
