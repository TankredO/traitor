import sys
from pathlib import Path
from typing import Optional

from cloup import (
    command,
    option,
    help_option,
    Path as PathType,
    option_group,
)

from ..defaults import DEFAULT_N_POLYGON_VERTICES


def run_single(
    image_file: Path,
    mask_file: Path,
    out_dir: Optional[Path],
    padding: int,
    n_vertices: int = DEFAULT_N_POLYGON_VERTICES,
):
    image_name = image_file.with_suffix("").name
    if out_dir is None:
        out_dir = Path(f"{image_name}_aligned")

    import numpy as np
    import pandas as pd
    import cv2
    import skimage.transform
    import skimage.morphology
    import pyefd

    from ..tools import (
        get_contours,
        resample_polygon,
        rotate_upright,
        align_shapes,
        extract_subimage,
    )

    # read in image and mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_file))

    # get contours from mask
    min_size = 1  # we assume only perfect masks
    contours = get_contours(mask, min_size=min_size)
    contours = [
        resample_polygon(c, n_points=n_vertices).astype(np.float32) for c in contours
    ]

    # align contours/polygons
    # the first contour is used as reference
    # TODO: let user select a contour as reference?
    ref, _ = rotate_upright(contours[0])
    ref_approx = pyefd.reconstruct_contour(
        pyefd.elliptic_fourier_descriptors(ref, order=3, normalize=False),
        num_points=100,
    )

    # Rotate all other contours to best match the reference contour
    # Shapes are approximated using elliptic Fourier analysis
    contours_rot = []
    rots = []
    for contour in contours:
        contour_approx = pyefd.reconstruct_contour(
            pyefd.elliptic_fourier_descriptors(contour, order=3, normalize=False),
            num_points=100,
        )
        _, best_i, rotation, scale, _ = align_shapes(ref_approx, contour_approx)
        contour_approx_rot = contour_approx.dot(rotation.T)
        disparity = np.sum(np.square(ref_approx - contour_approx_rot))

        rots.append(np.arctan2(rotation.T[0, 0], rotation.T[1, 0]) * 180 / np.pi)
        contour_rot = contour.dot(rotation.T)
        contour_rot = np.r_[
            contour_rot[
                best_i:,
            ],
            contour_rot[
                :best_i,
            ],
        ]
        contours_rot.append(contour_rot)

    # extract objects and rotate them
    out_dir_contours = out_dir / "contours"
    out_dir_contours.mkdir(parents=True, exist_ok=True)
    out_dir_extractions = out_dir / "extractions"
    out_dir_extractions.mkdir(parents=True, exist_ok=True)
    out_dir_masks = out_dir / "masks"
    out_dir_masks.mkdir(parents=True, exist_ok=True)

    for i, (contour, angle) in enumerate(zip(contours, rots)):
        out_file_contour = out_dir_contours / f"{image_name}_{i}_contour.csv"
        contour_rot = contours_rot[i]
        contour_rot = contour_rot - contour_rot.mean(axis=0)
        pd.DataFrame(contour_rot, columns=["x", "y"]).to_csv(
            out_file_contour, index=False
        )

        sub_image, sub_mask, *_ = extract_subimage(contour, image)

        sub_image_rot = skimage.transform.rotate(sub_image, angle, resize=True, order=1)

        sub_mask_rot = skimage.transform.rotate(sub_mask, angle, resize=True, order=0)
        sub_mask_rot = skimage.morphology.binary_dilation(sub_mask_rot)
        # sub_mask_rot = skimage.morphology.binary_erosion(sub_mask_rot)
        sub_mask_contours = sorted(
            get_contours(sub_mask_rot, 1), key=lambda x: x.shape[0], reverse=True
        )
        if len(sub_mask_contours) < 1:
            print(
                "WARNING: contour could not be found after alignment. "
                f"Please check {mask_file} and {image_file}."
            )
            continue

        contour = resample_polygon(sub_mask_contours[0], n_vertices)

        sub_image_rot, sub_mask_rot, *_ = extract_subimage(
            contour,
            sub_image_rot,
            padding=(padding, padding, padding, padding),
            remove_background=True,
        )

        out_file_image = out_dir_extractions / f"{image_name}_{i}.png"
        cv2.imwrite(str(out_file_image), sub_image_rot * 255)

        out_file_mask = out_dir_masks / f"{image_name}_{i}_mask.png"
        cv2.imwrite(str(out_file_mask), sub_mask_rot * 255)

        if len(sub_mask_contours) > 1:
            print(
                "WARNING: found multiple contours in aligned mask. "
                f"Please check {out_file_mask} and {out_file_image}."
            )


def single_wrapped(args):
    run_single(*args)


@command(
    "align",
    help="Align contours and extract rotated contours for multiple images.",
    no_args_is_help=True,
)
@option_group(
    "Required options",
    option(
        "-i",
        "--image_dir",
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help="Input image directory.",
        required=True,
    ),
    option(
        "-m",
        "--mask_dir",
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help="""
            Input mask directory or detection output directory. Mask files must
            be PNG files named <IMAGE_NAME>_mask.png; i.e., for an image file
            "image_1.jpg" the corresponding mask must be named "image_1_mask.png".
        """,
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
            a directory with the name "<IMAGE_DIR>_aligned" will be created in
            the current working directory ({Path(".").resolve()}).
        """,
        default=None,
        required=False,
    ),
)
@option_group(
    "Output options",
    option(
        "-p",
        "--padding",
        type=int,
        default=5,
        help="Padding around contours.",
        show_default=True,
    ),
)
@option(
    "-t",
    "--n_proc",
    default=1,
    help="""
        Number of parallel processes to run. Should not be set higher than
        the number of CPU cores.
    """,
    show_default=True,
)
@help_option("-h", "--help")
def align(
    image_dir: Path,
    mask_dir: Path,
    out_dir: Optional[Path],
    padding: int,
    n_proc: int,
):
    import numpy as np
    from ..defaults import IMAGE_EXTENSIONS

    image_dir_name = image_dir.name
    if out_dir is None:
        out_dir = Path(f"{image_dir_name}_aligned")

    image_extensions = IMAGE_EXTENSIONS
    image_files = np.array(
        [f for f in image_dir.glob("*") if f.suffix.lower() in image_extensions]
    )
    image_names = np.array([f.with_suffix("").name for f in image_files])

    # check if mask_dir is detection dir
    dir_content = [f.name for f in mask_dir.iterdir()]
    dirs_matching = np.array([(n in dir_content) for n in image_names])
    # mask_dir is detection output
    if dirs_matching.sum() > 0:
        msg = (
            f"NOTE: Found {dirs_matching.sum()} directories in mask_dir ({mask_dir}) matching "
            f"image file names in image_dir ({image_dir}): "
            "assuming mask_dir to be a detection output directory."
        )
        print(msg)
        if len(dirs_matching) != dirs_matching.sum():
            msg = (
                "WARNING: Could not find detection output for images\n\t"
                + "\n\t".join(sorted(image_names[~dirs_matching]))
            )
            print(msg)

        mask_files = np.array(
            [
                mask_dir / image_name / f"{image_name}_mask.png"
                for image_name in image_names[dirs_matching]
            ]
        )
        for mask_file in mask_files:
            if not mask_file.exists():
                sys.stderr.write(f"ERROR: mask file {mask_file} does not exist.\n")
                sys.exit(1)
        image_files = image_files[dirs_matching]

    # mask_dir is a simple directory containing mask images.
    else:
        mask_files = np.array(
            [mask_dir / f"{image_name}_mask.png" for image_name in image_names]
        )
        mask_files_matching = np.array([f.exists() for f in mask_files])
        if mask_files_matching.sum() < 1:
            sys.stderr.write(f"Could not find any mask in mask_dir ({mask_dir}).\n")
            sys.exit(1)
        if len(mask_files_matching) - mask_files_matching.sum() > 0:
            msg = "WARNING: Could not find mask files for images\n\t" + "\n\t".join(
                sorted(image_names[~mask_files_matching])
            )
            print(msg)
        mask_files = mask_files[mask_files_matching]
        image_files = image_files[mask_files_matching]

    # prepare arguments for parallel processing
    args_list = [
        (
            image_file,
            mask_file,
            out_dir.joinpath(image_file.with_suffix("").name),
            padding,
        )
        for image_file, mask_file in zip(image_files, mask_files)
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    # parallel runs
    import multiprocessing
    from tqdm import tqdm

    with multiprocessing.Pool(processes=n_proc) as pool:
        list(
            tqdm(
                pool.imap_unordered(single_wrapped, args_list),
                total=len(image_files),
            )
        )
