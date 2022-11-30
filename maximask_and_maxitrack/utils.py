"""File containing utility functions for MaxiMask and MaxiTrack inference."""
import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.interpolate as interp
import scipy.signal as sign
from astropy.io import fits


def timeit(f):
    """Decorator to measure execution time of a function.

    Args:
        f (function): function to decorate.
    Returns:
        timed (function): decorated function.
    """

    def timed(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        return result, round(time.perf_counter() - ts, 3)

    return timed


def str2bool(v):
    """Translate possible boolean inputs to boolean type.

    Args:
        v (string): value to translate to bool.
    Returns
        (bool): resulting bool.
    """

    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def read_config_file(file_path, nb_classes, log, to_float=False):
    """Reads a configuration file.

    Args:
        file_path (string): file path to read.
        nb_classes (int): number of classes.
        log (logging.log): log object.
        to_float (bool): whether to cast the read configuration parameters to float or bool.
    Returns:
        (np.ndarray): configuration parameters.
    """

    config_params = []
    try:
        with open(file_path) as fd:
            lines = fd.readlines()
    except IOError:
        log.error(f"Config file {file_path} not found")
        raise

    if len(lines) != nb_classes:
        log.error(f"Config file {file_path} does not have one row per class")
        raise ValueError

    for line in lines:
        if to_float:
            param = float(line.split()[1])
        else:
            param = int(line.split()[1])
        config_params.append(param)

    log.info(f"Using {file_path} config file")

    if to_float:
        return np.array(config_params, dtype=np.float32)
    else:
        return np.array(config_params, dtype=np.bool)


def get_file_list(im_path):
    """Gets the list of files to process.

    Args:
        im_path (string): path specifying the images to process.
    Returns:
        file_list (string): list of file to process.
    """

    file_list = []

    if os.path.isdir(im_path):
        # im_path is a directory
        # add all the fits files from the directory
        files = os.listdir(im_path)
        for f in files:
            if "fits" in f and "mask" not in f:
                fname = os.path.join(im_path, f)
                file_list.append(fname)
    elif os.path.isfile(im_path):
        # im_path is a file
        # check the extension
        _, im_path_ext = os.path.splitext(im_path)
        if im_path_ext == ".list":
            # im_path is a list file
            # add all the files from the list
            with open(im_path) as fd:
                lines = fd.readlines()
            for line in lines:
                fname = line.rstrip()
                if (
                    (os.path.isfile(fname) or "[" in fname)
                    and "fits" in fname
                    and "mask" not in fname
                ):
                    # fits file or specific fits file hdu
                    file_list.append(fname)
        elif im_path_ext == ".fits" or (
            im_path_ext in [".fz", ".gz"] and ".fits" in im_path
        ):
            # im_path is a fits file
            file_list.append(im_path)
        else:
            raise ValueError(
                f"Provided im_path <{im_path}> does not have a valid format"
            )
    elif "[" in im_path and "fits" in im_path and "mask" not in im_path:
        # im_path is a specific fits file hdu
        file_list.append(im_path)
    else:
        raise FileNotFoundError(f"Provided im_path <{im_path}> does not exist")

    return file_list


def check_hdu(hdu, min_size):
    """Checks whether an HDU is to be processed depending on its data shape and type.

    Args:
        hdu (astropy.io.fits.HDU): hdu to consider.
        min_size (int): minimal image size.
    Returns:
        (bool): whether the hdu is to be processed or not.
    """

    infos = hdu._summary()
    ds = infos[4]
    size_b = len(ds) == 2 and ds[0] > min_size and ds[1] > min_size
    dt = infos[5]
    data_type_b = (
        "float16" in dt
        or "float32" in dt
        or "float64" in dt
        or "int16" in dt
        or "int32" in dt
        or "int64" in dt
        or "uint32" in dt
    )

    return size_b and data_type_b, infos[2]


def background_est(im, k=3, mesh_size=200, full=True, pad=1):
    """Makes a Sextractor-like background estimation.
    Args:
        im (np.ndarray): input image.
        k (int): value to use for sigma-clipping.
        mesh_size (int): size of the local mesh/blocks.
        full (bool): whether to return local estimations or full continuous maps at image resolution.
        pad (int): how many blocks to pad around the image (use pad>1 can avoid interpolation artifacts on small images).
    Returns:
        back_v/back_map (list, np.ndarray) list of sky level local estimations or full map of the sky level.
        sig_v/sig_map (list, np.ndarray): list of sky sigma local estimations or full map of the sky sigma.
    """

    # get the number blocks to process
    h, w = im.shape
    h_blocks = (h - h % mesh_size) // mesh_size
    w_blocks = (w - w % mesh_size) // mesh_size
    x_l, y_l = [], []
    for y in range(h_blocks):
        for x in range(w_blocks):
            x_l.append((w % mesh_size) // 2 + x * mesh_size)
            y_l.append((h % mesh_size) // 2 + y * mesh_size)

    # process the blocks
    nb_blocks = len(x_l)
    back_v = [None] * nb_blocks
    sig_v = [None] * nb_blocks
    for b in range(nb_blocks):
        x_b, x_e = x_l[b], min(x_l[b] + mesh_size, w)
        y_b, y_e = y_l[b], min(y_l[b] + mesh_size, h)

        mesh = im[y_b:y_e, x_b:x_e]
        if np.all(mesh == 0):
            back_v[b] = 0
            sig_v[b] = 1
        else:
            # sigma clipping around the median
            cur_mask = np.ones_like(mesh)
            idx = np.where(cur_mask)
            cur_med = np.median(mesh[idx[0], idx[1]])
            init_std = np.std(mesh[idx[0], idx[1]])
            cur_std = init_std
            while True:
                to_keep = np.logical_and(
                    cur_mask * (mesh >= cur_med - k * cur_std),
                    cur_mask * (mesh <= cur_med + k * cur_std),
                )
                if np.all(to_keep == cur_mask):
                    break
                else:
                    cur_mask *= to_keep
                    idx = np.where(cur_mask)
                    cur_med = np.median(mesh[idx[0], idx[1]])
                    cur_std = np.std(mesh[idx[0], idx[1]])
            if cur_std > 0.8 * init_std and cur_std < 1.2 * init_std:
                back_v[b] = np.mean(mesh[idx[0], idx[1]])
            else:
                back_v[b] = 2.5 * cur_med - 1.5 * np.mean(mesh[idx[0], idx[1]])
            sig_v[b] = cur_std

    if not full:
        return back_v, sig_v
    else:
        back_map = build_map(back_v, h_blocks, w_blocks, x_l, y_l, mesh_size, h, w, pad)
        sig_map = build_map(sig_v, h_blocks, w_blocks, x_l, y_l, mesh_size, h, w, pad)
        return back_map, sig_map


def build_map(grid_v, h_blocks, w_blocks, x_l, y_l, mesh_size, h, w, pad):
    """Median filtering and interpolatation of local estimations.
    Args:
        grid_v (list): list of locally estimated values.
        h_blocks, w_blocks (int): number of mesh blocks for each axis.
        x_l, y_l (list): list of local estimation coordinates.
        mesh_size (int): size of the local mesh/blocks.
        h, w (int): size of the full image.
        pad (int): how many blocks to pad around the image (use pad>1 can avoid interpolation artifacts on small images).
    Returns:
        final_map (np.ndarray): full (h, w) interpolated map of the local estimations.
    """

    # set the values into the 2d grid to filter and interpolate
    grid = np.zeros([h_blocks, w_blocks])
    for b in range(len(grid_v)):
        grid[int(y_l[b] / mesh_size), int(x_l[b] / mesh_size)] = grid_v[b]

    # clean outliers
    grid = clean_grid(grid)

    # pad this grid to avoid interpolation artefacts on sides
    pad_grid = np.pad(grid, pad, "edge")

    # median filter
    pad_grid = sign.medfilt(pad_grid, 3)

    # interpolate the grid
    f = interp.RectBivariateSpline(
        np.arange(-pad * mesh_size / 2, (h_blocks + pad) * mesh_size + 1, mesh_size)
        + (h % mesh_size) // 2,
        np.arange(-pad * mesh_size / 2, (w_blocks + pad) * mesh_size + 1, mesh_size)
        + (w % mesh_size) // 2,
        pad_grid,
    )
    final_map = f(np.arange(h), np.arange(w)).astype(np.float32)

    return final_map


def clean_grid(grid):
    """Detects outliers in the grid and interpolate them.
    Args:
        grid (np.ndarray): input grid.
    Returns:
        grid (np.ndarray): grid with interpolated outliers.
    """

    # sigma clipping around the median
    cur_mask = np.ones_like(grid, dtype=np.int32)
    k = 0
    while True:
        idx = np.where(cur_mask)
        cur_med = np.median(grid[idx[0], idx[1]])
        cur_std = np.std(grid[idx[0], idx[1]])
        out_idx = np.where(
            np.logical_or(
                cur_mask * (grid > cur_med + 3 * cur_std),
                cur_mask * (grid < cur_med - 3 * cur_std),
            )
        )
        if len(out_idx[0]) == 0 or k == 5:
            break
        else:
            cur_mask[out_idx[0], out_idx[1]] = 0
        k += 1

    # interpolate outliers
    out_mask = np.logical_not(cur_mask)
    if np.any(out_mask):
        h, w = grid.shape
        np.place(grid, out_mask, float("nan"))
        while np.any(np.isnan(grid)):
            idx = np.where(out_mask)
            for k in range(len(idx[0])):
                y, x = idx[0][k], idx[1][k]
                tmp_mask = np.zeros_like(grid)
                for yy in range(y - 1, y + 2):
                    for xx in range(x - 1, x + 2):
                        if (
                            yy >= 0
                            and yy < h
                            and xx >= 0
                            and xx < w
                            and not math.isnan(grid[yy, xx])
                        ):
                            tmp_mask[yy, xx] = 1
                tmp_idx = np.where(tmp_mask)
                if len(tmp_idx[0]):
                    v = np.sum(grid[tmp_idx[0], tmp_idx[1]]) / np.sum(tmp_mask)
                    grid[y, x] = v

    return grid


@timeit
def image_norm(im):
    """Normalizes the image.

    Args:
        im (np.ndarray): input image.
    Returns:
        im (np.ndarray): normalized image.
    """

    # safety cast
    im = im.astype(np.float32)

    # preprocessing
    np.place(im, np.isnan(im), 0)
    np.place(im, im > 80000, 80000)
    np.place(im, im < -500, -500)

    # normalization
    bg_map, si_map = background_est(im)
    im -= bg_map
    im /= si_map

    return im
