import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple

from cloup import command, option, help_option, Path as PathType, Choice, option_group


def run_single(
    image_file: Path,
    out_dir: Optional[Path],
    background_color: str,
    masks_output: bool,
    bbox_output: bool,
    contour_output: bool,
    padding: int,
    rm_bg: bool,
    adaptive: bool,
):
    import warnings
    import numpy as np

    with warnings.catch_warnings(record=True) as w:
        image_name = image_file.with_suffix("").name

        # prepare output files
        if out_dir is None:
            out_dir = Path(image_name)
        out_file_mask = (out_dir / f"{image_name}_mask").with_suffix(".png")

        # parse background color string
        bg_col: Optional[Iterable[int]] = None
        if not background_color is None:
            background_color = "".join(background_color.split())
            if "," in background_color:
                try:
                    bg_col = [int(s) for s in background_color.split(",")][:3]
                except:
                    sys.stderr.write("ERROR: could not parse background_color.\n")
                    sys.exit(1)
            elif "#" in background_color:
                try:
                    bg_col = [b for b in bytes.fromhex(background_color[1:])]
                except:
                    sys.stderr.write("ERROR: could not parse background_color.\n")
                    sys.exit(1)
            else:
                sys.stderr.write("ERROR: invalid background_color option.\n")
                sys.exit(1)
            if any([x > 255 or x < 0 for x in bg_col]):
                sys.stderr.write("ERROR: color values must be in the range 0..255.\n")
                sys.exit(1)

        import skimage.io
        import cv2
        from ..tools import (
            segment_image_adaptive,
            get_minsize_adaptive,
            get_minsize_adaptive2,
            filter_bin_image,
            get_contours,
            resample_polygon,
            extract_subimage,
            segment_image2,
        )
        from ..defaults import (
            DEFAULT_N_POLYGON_VERTICES,
            DEFAULT_AREA_THRESHOLD,
            DEFAULT_HIGH_N_CONTOURS,
        )

        image = skimage.io.imread(image_file)[:, :, :3]  # ignore alpha channel

        if adaptive:
            bin_image = segment_image_adaptive(image, bg_col=bg_col, scale=None)
        else:
            bin_image = segment_image2(image, bg_col=bg_col, k=2)

        min_size = get_minsize_adaptive2(bin_image)
        area_threshold = (
            DEFAULT_AREA_THRESHOLD  # size of holes that will be filled within objects
        )
        bin_image = filter_bin_image(
            bin_image, min_size=min_size, area_threshold=area_threshold
        )

        contours = get_contours(bin_image=bin_image, min_size=min_size)

        n_vertices = (
            DEFAULT_N_POLYGON_VERTICES  # number of vertices to resample contours to
        )
        contours = [resample_polygon(c, n_vertices) for c in contours]

        out_dir.mkdir(parents=True, exist_ok=True)

        if masks_output:
            out_dir_masks = out_dir / "masks"
            out_dir_masks.mkdir(parents=True, exist_ok=True)

        if bbox_output:
            out_dir_extractions = out_dir / "extractions"
            out_dir_extractions.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_file_mask), bin_image * 255)

        if masks_output or bbox_output:
            for i, contour in enumerate(contours):
                sub_image, mask, bbox = extract_subimage(
                    contour,
                    image,
                    remove_background=rm_bg,
                    padding=(padding, padding, padding, padding),
                )

                if masks_output:
                    cv2.imwrite(
                        str(
                            (out_dir_masks / f"{image_name}_mask{i}").with_suffix(
                                ".png"
                            )
                        ),
                        mask * 255,
                    )
                if bbox_output:
                    cv2.imwrite(
                        str(
                            (out_dir_extractions / f"{image_name}_bbox{i}").with_suffix(
                                ".png"
                            )
                        ),
                        sub_image[:, :, [2, 1, 0]],
                    )

        if contour_output:
            out_file_contours = (out_dir / f"{image_name}_contours").with_suffix(".jpg")
            contours_cv2 = [c[:, [1, 0]].astype(np.int32) for c in contours]
            contour_image = image.copy()
            cv2.drawContours(contour_image, contours_cv2, -1, (255, 215, 0), 3)
            cv2.imwrite(str(out_file_contours), contour_image[:, :, [2, 1, 0]])

    if len(w) > 0:
        warnings_str = ", ".join([str(_w.message) for _w in w])
        warnings.warn(f"{image_file.name}: {warnings_str}")

    n_contours = len(contours)
    if n_contours > DEFAULT_HIGH_N_CONTOURS:
        print(
            f"WARNING: Found a high number of contours ({n_contours}). Please check {out_dir}."
        )


def single_wrapped(args):
    run_single(*args)


@command(
    "extract",
    help="Detect objects and extract their contours/masks from images.",
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
        help="Directory containing images in PNG format.",
        required=True,
    ),
    option(
        "-o",
        "--out_dir",
        type=PathType(
            file_okay=False, dir_okay=True, resolve_path=True, path_type=Path
        ),
        help=f"""
            A directory with the specified name will be created if it does not exist.
            By default the directory will be created in the current working directory ({Path(".").resolve()}).
        """,
        default=None,
        required=True,
    ),
)
@option(
    "-c",
    "--background_color",
    type=str,
    default=None,
    help="""
        Background color. Can be passed as comma-separated list of RGB values
        (e.g., "12, 52, 128") or alternatively as hex color code (e.g., "#04245f").
        By default the most common color range will be assumed to be the background
        color.
    """,
)
@option_group(
    "Output options",
    option(
        "-m",
        "--masks_output",
        is_flag=True,
        help="Generate mask output for every detected object.",
    ),
    option(
        "-b",
        "--bbox_output",
        is_flag=True,
        help="Extract bounding boxes for every detected object.",
    ),
    option(
        "-u",
        "--contour_output",
        is_flag=True,
        help="""
            Generate additional output image with contours to easily
            check the quality of image segmentation.""",
    ),
    option(
        "-p",
        "--padding",
        type=int,
        default=5,
        help="Padding around objects for mask and bounding box outputs.",
        show_default=True,
    ),
    option(
        "--rm_bg",
        is_flag=True,
        help="Set background pixel in bbox output to 0 (black).",
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
)
@option(
    "--adaptive",
    is_flag=True,
    help="EXPERIMENTAL: use adaptive segmentation algorithm",
)
@help_option("-h", "--help")
def extract(
    image_dir: Path,
    out_dir: Optional[Path],
    background_color: Optional[str],
    masks_output: bool,
    bbox_output: bool,
    contour_output: bool,
    padding: int,
    rm_bg: bool,
    n_proc: int,
    adaptive: bool,
):
    import sys
    import multiprocessing
    from tqdm import tqdm
    from ..defaults import IMAGE_EXTENSIONS

    image_dir_name = image_dir.name
    if out_dir is None:
        out_dir = Path(f"{image_dir_name}_detections")

    image_extensions = IMAGE_EXTENSIONS
    image_files = [
        f for f in image_dir.glob("*") if f.suffix.lower() in image_extensions
    ]
    if len(image_files) < 1:
        sys.stderr.write(f"ERROR: Could not find any image in {image_dir}.")
        sys.exit(1)

    args_list = [
        (
            f,
            out_dir.joinpath(f.with_suffix("").name),
            background_color,
            masks_output,
            bbox_output,
            contour_output,
            padding,
            rm_bg,
            adaptive,
        )
        for f in image_files
    ]
    out_dir.mkdir(parents=True, exist_ok=True)

    with multiprocessing.Pool(processes=n_proc) as pool:
        list(
            tqdm(
                pool.imap_unordered(single_wrapped, args_list),
                total=len(image_files),
            )
        )
