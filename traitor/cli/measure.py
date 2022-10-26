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


from ..defaults import DEFAULT_N_CONTOUR_VERTICES


def run_single(
    image_file: Path,
    mask_file: Path,
    contour_file: Path,
    n_colors: int = 4,
    n_vertices: int = DEFAULT_N_CONTOUR_VERTICES,
):
    import warnings

    warnings.filterwarnings(action="ignore", category=RuntimeWarning)

    image_name = image_file.with_suffix("").name

    import numpy as np
    import pandas as pd
    import cv2
    import skimage.transform
    import skimage.measure
    import skimage.morphology
    import skimage.color
    import skimage.feature
    import skimage.exposure
    from ..tools import (
        get_contours,
        resample_polygon,
        reset_starting_point,
    )
    from ..median_cut import median_cut, dominant_colors_mc, dominant_color_image_mc

    # read in image and mask
    mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
    image = cv2.imread(str(image_file))

    # # get contours from mask
    # n_vertices = DEFAULT_N_POLYGON_VERTICES
    # min_size = 1  # we assume only perfect masks
    # try:
    #     contour = get_contours(mask, min_size=min_size)[0]
    #     contour = resample_polygon(contour, n_points=n_vertices)
    # except:
    #     print(f'WARNING: no contour found in {mask_file}')

    # get contour from contour file
    contour = pd.read_csv(contour_file).values
    # set first contour point to lower middle
    contour = reset_starting_point(contour)
    # resample contour
    contour = resample_polygon(contour, n_vertices)
    # "standardize" contour (for size invariance)
    contour = (contour - contour.mean(0)) / (contour[:, 1].max() - contour[:, 1].min())

    # == shape
    props = skimage.measure.regionprops(mask)[0]

    bbox = props.bbox
    length = bbox[2] - bbox[0]
    # width = bbox[3] - bbox[1]
    # length = np.max(np.sum(mask == 255, axis=0))
    width = np.max(np.sum(mask == 255, axis=1))
    aspect_ratio = width / length

    area = props.area
    perimeter = props.perimeter

    chull = skimage.morphology.convex_hull_image(mask) * 255
    chull_props = skimage.measure.regionprops(chull)[0]
    chull_perimeter = chull_props.perimeter
    chull_area = chull_props.area

    surface_structure = chull_perimeter / perimeter
    solidity = area / chull_area

    circularity = 4 * np.pi * area / np.power(perimeter, 2)

    # == dominant colors
    def build_pc_dict(
        colors,
        counts,
        prefix,
        col_comp_names,
    ):
        if np.issubdtype(counts[0], int):
            suffix = "count"
        else:
            suffix = "frac"

        pc_dict = {}
        for i, col in enumerate(colors):
            for val, c in zip(col, col_comp_names):
                pc_dict[f"{prefix}_{i}_{c}"] = val
        for i, count in enumerate(counts):
            pc_dict[f"{prefix}_{i}_{suffix}"] = count

        return pc_dict

    # dominant color (RGB)
    # colors_rgb, counts_rgb = primary_colors(image[:, :, [2, 1, 0]], mask, n_colors)
    depth = 0
    tmp = n_colors
    while tmp > 1:
        tmp = tmp // 2
        depth += 1

    cl_rgb = median_cut(image=image[:, :, [2, 1, 0]], mask=mask, depth=depth)
    colors_rgb, counts_rgb = dominant_colors_mc(image[2, 1, 0], cl_rgb)
    colors_rgb = np.round(colors_rgb, 0).astype(np.uint8)
    frac_rgb = counts_rgb / counts_rgb.sum()
    colors_rgb_dict = build_pc_dict(colors_rgb, frac_rgb, "rgb", ("r", "g", "b"))

    # median color (RGB)
    # remove outer pixels to reduce color noise (reflected background color)
    kernel = np.ones((5, 5), np.uint8)
    mask2 = cv2.erode(mask, kernel)

    pixel_values_rgb = skimage.exposure.adjust_gamma(
        image[mask2 != 0][:, [2, 1, 0]], gamma=2.2
    )
    rgb_median = np.median(pixel_values_rgb, 0)
    rgb_median_dict = {
        "R_median": rgb_median[0],
        "G_median": rgb_median[1],
        "B_median": rgb_median[2],
    }

    rgb_mean = np.mean(pixel_values_rgb, 0)
    rgb_mean_dict = {
        "R_mean": rgb_mean[0],
        "G_mean": rgb_mean[1],
        "B_mean": rgb_mean[2],
    }

    pixel_values_srgb = image[mask2 != 0][:, [2, 1, 0]]
    srgb_median = np.median(pixel_values_srgb, 0)
    srgb_median_dict = {
        "sR_median": srgb_median[0],
        "sG_median": srgb_median[1],
        "sB_median": srgb_median[2],
    }

    srgb_mean = np.mean(pixel_values_srgb, 0)
    srgb_mean_dict = {
        "sR_mean": srgb_mean[0],
        "sG_mean": srgb_mean[1],
        "sB_mean": srgb_mean[2],
    }

    measurements = pd.DataFrame(
        dict(
            image_name=image_name,
            image_file=str(image_file),
            mask_file=str(mask_file),
            length=length,
            width=width,
            aspect_ratio=aspect_ratio,
            area=area,
            perimeter=perimeter,
            surface_structure=surface_structure,
            solidity=solidity,
            circularity=circularity,
            **rgb_median_dict,
            **rgb_mean_dict,
            **srgb_median_dict,
            **srgb_mean_dict,
            **colors_rgb_dict,
        ),
        index=[image_name],
    )

    contour_df = pd.DataFrame.from_records(
        [contour.flatten()],
        columns=[
            f"{c}_{i}"
            for c, i in zip(
                ["x", "y"] * contour.shape[0], np.repeat(np.arange(contour.shape[0]), 2)
            )
        ],
        index=[image_name],
    )

    measurements = pd.concat([measurements, contour_df], axis=1)

    return measurements


def single_wrapped(args):
    return run_single(*args[0:-1]), args[-1]


@command(
    "measure",
    help="Calculate measurements for multiple image-mask pairs.",
    no_args_is_help=True,
)
@option_group(
    "Required options",
    option(
        "-i",
        "--input_dir",
        type=PathType(
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
            path_type=Path,
        ),
        help="""
            Directory containing the outputs of the "traitor align" command.
        """,
        required=True,
    ),
    option(
        "-o",
        "--out_file",
        type=PathType(
            file_okay=True, dir_okay=False, resolve_path=True, path_type=Path
        ),
        help=f"""
            Output (CSV) file. By default a file with the name "<INPUT_DIR>_measurements.csv"
            will be created in the current working directory ({Path(".").resolve()}).
        """,
        default=None,
        required=False,
    ),
)
@option_group(
    "Color options",
    option(
        "-n",
        "--n_colors",
        type=int,
        default=4,
        help="Number of dominant colors to extract. The specified value should be power of two.",
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
def measure(
    input_dir: Path,
    out_file: Optional[Path],
    n_colors: int,
    n_proc: int,
    n_vertices: int = DEFAULT_N_CONTOUR_VERTICES,
):
    import numpy as np
    from ..defaults import IMAGE_EXTENSIONS

    if out_file is None:
        input_dir_name = input_dir.name
        out_file = Path(f"{input_dir_name}_measurements.csv")

    print("Looking for files ...")
    dirs = [f for f in input_dir.iterdir() if f.is_dir() and not f.name.startswith(".")]
    image_files = np.array([])
    mask_files = np.array([])
    contour_files = np.array([])
    groups = np.array([])
    for d in dirs:
        img_dir = d / "extractions"
        mask_dir = d / "masks"
        contour_dir = d / "contours"

        cur_img_files = np.array(list(img_dir.glob("*.png")))
        cur_image_names = np.array([f.with_suffix("").name for f in cur_img_files])

        cur_mask_files = np.array([mask_dir / f"{n}_mask.png" for n in cur_image_names])
        cur_mask_files_matching = np.array([f.exists() for f in cur_mask_files])
        if cur_mask_files_matching.sum() < 1:
            sys.stderr.write(f"Could not find any mask in mask_dir ({mask_dir}).\n")
            sys.exit(1)
        if len(cur_mask_files_matching) - cur_mask_files_matching.sum() > 0:
            msg = "WARNING: Could not find mask files for images\n\t" + "\n\t".join(
                sorted(cur_image_names[~cur_mask_files_matching])
            )
            print(msg)

        cur_contour_files = np.array(
            [contour_dir / f"{n}_contour.csv" for n in cur_image_names]
        )
        cur_contour_files_matching = np.array([f.exists() for f in cur_contour_files])
        if cur_contour_files_matching.sum() < 1:
            sys.stderr.write(
                f"Could not find any contour in contour_dir ({contour_dir}).\n"
            )
            sys.exit(1)
        if len(cur_contour_files_matching) - cur_contour_files_matching.sum() > 0:
            msg = "WARNING: Could not find contour files for images\n\t" + "\n\t".join(
                sorted(cur_image_names[~cur_contour_files_matching])
            )
            print(msg)

        image_files = np.append(
            image_files,
            cur_img_files[cur_mask_files_matching & cur_contour_files_matching],
        )
        mask_files = np.append(
            mask_files,
            cur_mask_files[cur_mask_files_matching & cur_contour_files_matching],
        )
        contour_files = np.append(
            contour_files,
            cur_contour_files[cur_mask_files_matching & cur_contour_files_matching],
        )
        groups = np.append(
            groups,
            [
                d.name
                for _ in range(
                    (cur_mask_files_matching & cur_contour_files_matching).sum()
                )
            ],
        )
    # prepare arguments for parallel processing
    args_list = [
        (
            image_file,
            mask_file,
            contour_file,
            n_colors,
            group,  # need to pass groups since we are using imap_unordered for parallel processing
        )
        for image_file, mask_file, contour_file, group in zip(
            image_files, mask_files, contour_files, groups
        )
    ]

    # parallel runs
    import multiprocessing
    import pandas as pd
    from tqdm import tqdm

    with multiprocessing.Pool(processes=n_proc) as pool:
        if out_file.exists():
            out_file.unlink()

        for measurements, group in tqdm(
            pool.imap_unordered(single_wrapped, args_list),
            total=len(image_files),
        ):
            measurements["group"] = group
            measurements.insert(3, "group", measurements.pop("group"))
            measurements.to_csv(
                out_file, index=False, mode="a", header=not out_file.exists()
            )

    print("Checking output ...")
    # Test read output
    measurements = pd.read_csv(out_file)
    # reshape [n, vertices, xy]
    contours = measurements.iloc[:, -n_vertices * 2 :].values.reshape(-1, n_vertices, 2)

    # check if NaN in contours. NaN values indicate that the extraction was not successful
    nan_row_idcs = np.array([np.any(np.isnan(c)) for c in contours])
    if sum(nan_row_idcs) > 0:
        nan_measurements: pd.DataFrame = measurements.iloc[nan_row_idcs]
        nan_groups = nan_measurements["group"].unique()
        msg = f"WARNING: Encountered NaN. Please check extractions for the following groups:\n"
        msg = msg + "\n".join(nan_groups)
        sys.stderr.write(msg)
