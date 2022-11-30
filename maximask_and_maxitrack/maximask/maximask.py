"""Script to run MaxiMask inference."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import argparse
import logging as log
import math
import os
import sys
import time

import numpy as np
import tqdm
from astropy.io import fits

from maximask_and_maxitrack import utils


class MaxiMask_inference(object):
    """Class to run MaxiMask inference."""

    nb_classes = 14
    im_size = 400
    im2 = im_size // 2
    im4 = im_size // 4
    tr_priors = np.array(
        [
            0.010,
            0.009,
            0.036,
            0.001,
            0.001,
            0.010,
            0.103,
            0.126,
            0.171,
            0.002,
            0.044,
            0.118,
            0.126,
            0.395,
        ],
        dtype=np.float32,
    )
    class_names = [
        "CR: Cosmic Rays",
        "HCL: Hot Columns/Lines",
        "DCL: Dead Columns/Lines/Clusters",
        "HP: Hot Pixels",
        "DP: Dead Pixels",
        "P: Persistence",
        "TRL: TRaiLs",
        "FR: residual FRinging",
        "NEB: NEBulosities",
        "SAT: SATurated pixels",
        "SP: diffraction SPikes",
        "OV: Overscan",
        "BBG: Bright BackGround",
        "BG: BackGround",
    ]
    class_abbr = [
        "CR",
        "HCL",
        "DCL",
        "HP",
        "DP",
        "P",
        "TRL",
        "FR",
        "NEB",
        "SAT",
        "SP",
        "OV",
        "BBG",
        "BG",
    ]

    def __init__(
        self, im_path, net_dir, class_flags, priors, thresholds, sing_mask, batch_size
    ):
        """Initializes the MaxiMask_inference class.

        Args:
            im_path (string): path to the images to be processed. This can be a fits file, a directory or a list file.
            net_dir (string): path to the MaxiMask neural network directory.
            class_flags (np.ndarray): boolean flags for class selection.
            priors (np.ndarray): priors for each class. None if the prior modification is not requested.
            thresholds (np.ndarray): thresholds for each class. None if thresholding is not requested.
            sing_mask (bool): boolean indicating if the single binary output map option is requested.
            batch_size (int): batch size to use for inference.
        """

        self.im_path = im_path
        self.net_dir = net_dir

        # builds the class indexes of the requested classes
        self.class_flags = class_flags
        class_idx = []
        for k in range(self.nb_classes):
            if self.class_flags[k]:
                class_idx.append(k)
        self.class_idx = np.array(class_idx, dtype=np.int32)

        # builds the prior factor values to be given for inference
        self.priors = priors
        if self.priors is not None:
            self.prior_factors = (
                self.tr_priors[self.class_flags]
                / (1 - self.tr_priors[self.class_flags])
            ) * ((1 - self.priors[self.class_flags]) / self.priors[self.class_flags])
        else:
            self.prior_factors = np.zeros([self.nb_classes], dtype=np.float32)

        # builds the threshold values to be given for inference
        self.thresholds = thresholds
        if self.thresholds is not None:
            self.thresh_inf = self.thresholds[self.class_flags]
        else:
            self.thresh_inf = np.zeros([self.nb_classes], dtype=np.float32)

        # builds the binary mask values to be given for inference
        self.sing_mask = sing_mask
        if sing_mask:
            self.bin_powers = np.power(
                2 * np.ones([self.nb_classes]), np.arange(self.nb_classes)
            )[self.class_flags].astype(np.float32)
            # if BG class is requested do not consider it for single mask option
            if self.nb_classes - 1 in self.class_idx:
                self.bin_powers[-1] = 0
        else:
            self.bin_powers = np.zeros([self.nb_classes], dtype=np.float32)

        self.batch_size = batch_size

    @utils.timeit
    def process_all(self):
        """Processes all the requested images.
        This will write the corresponding mask files on disk.
        """

        # get list of files to process
        file_list = utils.get_file_list(self.im_path)
        log.info(f"List of files to process: {file_list}")

        if len(file_list):
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            log.info("")
            log.info("##### Beginning of possible Tensorflow logs")
            import tensorflow as tf

            tf_model = tf.saved_model.load(self.net_dir)
            log.info("##### End of Tensorflow logs")
            log.info("")
            log.info(f"Using TensorFlow version {tf.__version__}")
            gpu_devices = tf.config.list_logical_devices("GPU")
            log.info(f"TensorFlow has created {len(gpu_devices)} logical GPU device(s)")

            # process each file of file
            for file_name in tqdm.tqdm(file_list, desc="ALL FILES"):
                log.info(f"Starting {file_name}")
                self.process_file(file_name, tf_model)
                log.info(f"{file_name} done")
        else:
            log.error(f"No file to process given input path <{self.im_path}>")

    def process_file(self, file_name, tf_model):
        """Processes a given fits file.
        This will write the corresponding mask file on disk.

        Args:
            file_name (string): name of the file to process.
            tf_model (tf.keras.Model): MaxiMask tensorflow model.
        """

        final_file_hdu = fits.HDUList()

        # make hdu tasks
        hdu_task_list = self.make_hdu_tasks(file_name)

        # check that there is at least one HDU to process
        at_least_one = False
        for _, task, _, _ in hdu_task_list:
            if task == "process":
                at_least_one = True

        # if at least one HDU to be processed
        if at_least_one:
            # go through all all HDUs
            for hdu_idx, task, hdu_type, hdu_shape in hdu_task_list:
                log.info(f"HDU {hdu_idx}: {task}")

                # get raw predictions
                hdu_preds, t = self.process_hdu(file_name, hdu_idx, task, tf_model)
                if hdu_shape is not None:
                    log.info(
                        f"Whole processing time (incl. preprocessing): {t:.2f}s, {np.prod(hdu_shape)/(t*1e06):.2f}MPix/s"
                    )

                # append the HDU
                if hdu_type == "PrimaryHDU":
                    hdu = fits.PrimaryHDU(hdu_preds)
                elif hdu_type == "ImageHDU" or hdu_type == "CompImageHDU":
                    hdu = fits.ImageHDU(hdu_preds)
                self.fill_header(hdu)
                final_file_hdu.append(hdu)

            # write file
            out_file_name = file_name
            if ".fz" in out_file_name:
                out_file_name = out_file_name.replace(".fz", "")
            if ".gz" in file_name:
                out_file_name = out_file_name.replace(".gz", "")
            if "[" in out_file_name:
                spec_hdu_idx = int(file_name.split("[")[1].split("]")[0])
                out_file_name = out_file_name.split("[")[0].replace(
                    ".fits", f".mask{spec_hdu_idx}.fits"
                )
                final_file_hdu.writeto(out_file_name, overwrite=True)
            else:
                final_file_hdu.writeto(
                    out_file_name.replace(".fits", ".mask.fits"), overwrite=True
                )
        else:
            log.info(f"Skipping {file_name} because no HDU was found to be processed")

    def make_hdu_tasks(self, file_name):
        """Makes the hdu task list for the given file to process.

        Args:
            file_name (string): name of the file to process.
        Returns:
            hdu_task_list (list): list of hdu tasks for the file to process.
                                  each element contains the hdu number, whether to "copy" or "process" and the type and shape of the data contained in the HDU.
        """

        hdu_task_list = []

        _, im_path_ext = os.path.splitext(file_name)
        if "[" in im_path_ext:
            # it is a specified HDU
            spec_hdu_idx = int(file_name.split("[")[1].split("]")[0])
            with fits.open(file_name.split("[")[0]) as file_hdu:
                specified_hdu = file_hdu[spec_hdu_idx]
                check, hdu_type = utils.check_hdu(specified_hdu, self.im_size)
                if check:
                    hdu_shape = specified_hdu.data.shape
                    hdu_task_list.append([spec_hdu_idx, "process", hdu_type, hdu_shape])
                else:
                    hdu_task_list.append([spec_hdu_idx, "copy", hdu_type, None])
        else:
            with fits.open(file_name) as file_hdu:
                nb_hdu = len(file_hdu)
                for k in range(nb_hdu):
                    check, hdu_type = utils.check_hdu(file_hdu[k], self.im_size)
                    if check:
                        hdu_shape = file_hdu[k].data.shape
                        hdu_task_list.append([k, "process", hdu_type, hdu_shape])
                    else:
                        hdu_task_list.append([k, "copy", hdu_type, None])

        return hdu_task_list

    @utils.timeit
    def process_hdu(self, file_name, hdu_idx, task, tf_model):
        """Processes the hdu of a given file according to the task.

        Args:
            file_name (string): name of the file to process.
            hdu_idx (int): index of the HDU to process.
            task (list): task related to the HDU to process (see make_hdu_tasks doc).
            tf_model (tf.keras.Model): MaxiMask tensorflow model.
        Returns:
            preds or hdu_data (np.ndarray): MaxiMask predictions for the requested HDU or same HDU data if the task is "copy".
        """

        # make file name
        _, im_path_ext = os.path.splitext(file_name)
        if "[" in im_path_ext:
            file_name = file_name.split("[")[0]

        # get input data
        with fits.open(file_name) as file_hdu:
            hdu = file_hdu[hdu_idx]
            hdu_data = hdu.data

        if task == "copy":
            return hdu_data
        elif task == "process":

            # prediction array
            h, w = hdu_data.shape
            if np.all(hdu_data == 0):
                return np.zeros_like(hdu_data, dtype=np.uint8)
            else:
                if self.sing_mask:
                    preds = np.zeros([h, w], dtype=np.int16)
                elif self.thresholds is not None:
                    preds = np.zeros([h, w, np.sum(self.class_flags)], dtype=np.uint8)
                else:
                    preds = np.zeros([h, w, np.sum(self.class_flags)], dtype=np.float32)

                # get list of block coordinate to process
                block_coord_list = self.get_block_coords(h, w)

                # preprocessing
                log.info("Preprocessing...")
                hdu_data, t = utils.image_norm(hdu_data)
                log.info(f"Preprocessing done in {t:.2f}s, {h*w/(t*1e06):.2f}MPix/s")

                # process all the blocks by batches
                # the process_batch method writes the predictions in preds by reference
                nb_blocks = len(block_coord_list)
                if nb_blocks <= self.batch_size:
                    # only one (possibly not full) batch to process
                    self.process_batch(hdu_data, preds, tf_model, block_coord_list)
                else:
                    # several batches to process + one last possibly not full
                    nb_batch = nb_blocks // self.batch_size
                    for b in tqdm.tqdm(range(nb_batch), desc="INFERENCE: "):
                        batch_coord_list = block_coord_list[
                            b * self.batch_size : (b + 1) * self.batch_size
                        ]
                        self.process_batch(hdu_data, preds, tf_model, batch_coord_list)
                    rest = nb_blocks - nb_batch * self.batch_size
                    if rest:
                        batch_coord_list = block_coord_list[-rest:]
                        self.process_batch(hdu_data, preds, tf_model, batch_coord_list)

                if not self.sing_mask:
                    preds = np.transpose(preds, (2, 0, 1))

                return preds

    def get_block_coords(self, h, w):
        """Gets the coordinate list of blocks to process.

        Args:
            h (int): full height of the image to process.
            w (int): full width of the image to process.
        Returns:
            coord_list (list): list of coordinates of blocks to process.
        """

        coord_list = []

        for y in range(0, h - self.im_size + 1, self.im2):
            for x in range(0, w - self.im_size + 1, self.im2):
                coord_list.append([x, y])
        if h % self.im_size:
            for x in range(0, w - self.im_size + 1, self.im2):
                coord_list.append([x, h - self.im_size])
        if w % self.im_size:
            for y in range(0, h - self.im_size + 1, self.im2):
                coord_list.append([w - self.im_size, y])
        if w % self.im_size and h % self.im_size:
            coord_list.append([w - self.im_size, h - self.im_size])

        return coord_list

    def process_batch(self, im_data, out_array, tf_model, batch_coord_list):
        """Processes a batch of inputs.
        This fills the output array by reference (class selection, application of priors and thresholds and single mask process happen within the tensorflow model).

        Args:
            im_data (np.ndarray): image data to process.
            out_array (np.ndarray): output array to fill with predictions.
            tf_model (tf.keras.Model): MaxiMask tensorflow model.
            batch_coord_list (list): list of coordinates of block to process for this batch.
        """

        # prepare input data batch
        h, w = im_data.shape
        inp = np.zeros(
            [len(batch_coord_list), self.im_size, self.im_size], dtype=np.float32
        )
        b = 0
        for x, y in batch_coord_list:
            inp[b] = im_data[y : y + self.im_size, x : x + self.im_size]
            b += 1
        inp = np.expand_dims(inp, axis=3)

        # make predictions
        res = tf_model(
            [inp, self.class_idx, self.prior_factors, self.thresh_inf, self.bin_powers],
            False,
        )

        # fill output array
        b = 0
        for x, y in batch_coord_list:
            # centered predictions
            out_array[
                self.im4 + y : y + self.im_size - self.im4,
                self.im4 + x : x + self.im_size - self.im4,
            ] = res[
                b,
                self.im4 : self.im_size - self.im4,
                self.im4 : self.im_size - self.im4,
            ]
            # ides if we are on the side of the CCD
            if x == 0:
                out_array[
                    self.im4 + y : y + self.im_size - self.im4, x : x + self.im4
                ] = res[b, self.im4 : self.im_size - self.im4, : self.im4]
            if y == 0:
                out_array[
                    y : y + self.im4, self.im4 + x : x + self.im_size - self.im4
                ] = res[b, : self.im4, self.im4 : self.im_size - self.im4]
            if x == w - self.im_size:
                out_array[
                    self.im4 + y : y + self.im_size - self.im4,
                    x + self.im_size - self.im4 : x + self.im_size,
                ] = res[
                    b,
                    self.im4 : self.im_size - self.im4,
                    self.im_size - self.im4 : self.im_size,
                ]
            if y == h - self.im_size:
                out_array[
                    y + self.im_size - self.im4 : y + self.im_size,
                    self.im4 + x : x + self.im_size - self.im4,
                ] = res[
                    b,
                    self.im_size - self.im4 : self.im_size,
                    self.im4 : self.im_size - self.im4,
                ]
            if x == 0 and y == 0:
                out_array[y : y + self.im4, x : x + self.im4] = res[
                    b, : self.im4, : self.im4
                ]
            if x == 0 and y == h - self.im_size:
                out_array[
                    y + self.im_size - self.im4 : y + self.im_size, x : x + self.im4
                ] = res[b, self.im_size - self.im4 : self.im_size, : self.im4]
            if x == w - self.im_size and y == 0:
                out_array[
                    y : y + self.im4, x + self.im_size - self.im4 : x + self.im_size
                ] = res[b, : self.im4, self.im_size - self.im4 : self.im_size]
            if x == w - self.im_size and y == h - self.im_size:
                out_array[
                    y + self.im_size - self.im4 : y + self.im_size,
                    x + self.im_size - self.im4 : x + self.im_size,
                ] = res[
                    b,
                    self.im_size - self.im4 : self.im_size,
                    self.im_size - self.im4 : self.im_size,
                ]
            b += 1

    def fill_header(self, hdu):
        """Fills the header of the given HDU with processing information.

        Args:
            hdu (astropy.io.fits.HDUList): HDU to make header for.
        """

        hdu.header["MM_UTC"] = time.asctime(time.gmtime())
        hdu.header.comments["MM_UTC"] = "MaxiMask UTC processing date"
        hdu.header["MM_LOC"] = time.asctime(time.localtime())
        hdu.header.comments["MM_LOC"] = "MaxiMask LOC processing date"

        if self.priors is not None:
            hdu.header["PRIORS"] = "Yes"
        else:
            hdu.header["PRIORS"] = "No"
        hdu.header.comments["PRIORS"] = "Whether prior modification was applied or not"

        if self.thresholds is not None:
            hdu.header["THRESH"] = "Yes"
        else:
            hdu.header["THRESH"] = "No"
        hdu.header.comments["THRESH"] = "Whether thresholding was applied or not"

        if self.sing_mask:
            hdu.header["SMASK"] = "Yes"
        else:
            hdu.header["SMASK"] = "No"
        hdu.header.comments["SMASK"] = "Whether binary power mask was applied or not"

        k = 0
        for cl in range(self.nb_classes):
            if self.class_flags[cl]:
                hdu.header[self.class_abbr[cl]] = self.class_names[cl]
                hdu.header[f"{self.class_abbr[cl]}_DIM"] = f"{k}"
                if self.priors is not None:
                    hdu.header[f"{self.class_abbr[cl]}_PR"] = f"{self.priors[cl]:.2f}"
                    hdu.header.comments[
                        f"{self.class_abbr[cl]}_PR"
                    ] = f"{self.class_abbr[cl]} prior"
                if self.thresholds is not None:
                    hdu.header[
                        f"{self.class_abbr[cl]}_TH"
                    ] = f"{self.thresholds[cl]:.2f}"
                    hdu.header.comments[
                        f"{self.class_abbr[cl]}_TH"
                    ] = f"{self.class_abbr[cl]} threshold"
                if self.sing_mask:
                    hdu.header[f"{self.class_abbr[cl]}_SM"] = f"{self.bin_powers[k]}"
                    hdu.header.comments[
                        f"{self.class_abbr[cl]}_SM"
                    ] = f"{self.class_abbr[cl]} binary power value"
                k += 1


def main():

    ### parameter parser
    parser = argparse.ArgumentParser(description="MaxiMask command line parameters:")

    # positional parameter
    parser.add_argument(
        "im_path", type=str, help="path to the image(s) to be processed"
    )

    # optional parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_dir = os.path.join(script_dir, "model")
    config_dir = os.path.join(script_dir, "config")
    parser.add_argument(
        "--net_dir",
        type=str,
        help="neural network graphs and weights directory. Default is </path_to_root/maximask_and_maxitrack/maximask/model>",
        default=net_dir,
    )
    parser.add_argument(
        "--config_dir",
        type=str,
        help="configuration file directory. Default is </path_to_root/maximask_and_maxitrack/maximask/config>",
        default=config_dir,
    )
    parser.add_argument(
        "--prior_modif",
        type=utils.str2bool,
        help="bool indicating if probability maps should be prior modified. Default is True",
        default=True,
    )
    parser.add_argument(
        "--proba_thresh",
        type=utils.str2bool,
        help="bool indicating if probability maps should be thresholded. Default is True",
        default=True,
    )
    parser.add_argument(
        "--single_mask",
        type=utils.str2bool,
        help="bool indicating if resulting masks are joined in a single mask using a binary base. Default is False",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="neural network batch size. Default is 8. You might want to use a lower value if you have RAM issues",
        default=8,
    )
    parser.add_argument(
        "-v", "--verbose", help="activate output verbosity", action="store_true"
    )

    # retrieve parameters
    args = parser.parse_args()
    im_path = args.im_path
    net_dir = args.net_dir
    config_dir = args.config_dir
    prior_modif = args.prior_modif
    proba_thresh = args.proba_thresh
    sing_mask = args.single_mask
    batch_size = args.batch_size
    verbose = args.verbose
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    nb_classes = 14

    # read flag file
    flag_file_path = os.path.join(config_dir, "classes.flags")
    class_flags = utils.read_config_file(
        flag_file_path, nb_classes, log
    )  # replace with file_path ?
    if np.all(class_flags == 0):
        log.error("No output class was requested...")
        raise ValueError

    # read prior file
    prior_file_path = os.path.join(config_dir, "classes.priors")
    if prior_modif:
        priors = utils.read_config_file(prior_file_path, nb_classes, log, to_float=True)
    else:
        priors = None

    # read thresh file
    thresh_file_path = os.path.join(config_dir, "classes.thresh")
    if proba_thresh:
        thresholds = utils.read_config_file(
            thresh_file_path, nb_classes, log, to_float=True
        )
    else:
        thresholds = None
    if thresholds is None and sing_mask:
        log.error("Cannot output single mask binary map if not thresholding")
        raise ValueError

    # build MaxiMask_inference object
    mm_inf = MaxiMask_inference(
        im_path, net_dir, class_flags, priors, thresholds, sing_mask, batch_size
    )

    # process all
    _, t = mm_inf.process_all()
    if t < 60:
        log.info(f"All done: {t:.2f}s")
    else:
        if t < 3600:
            m, s = divmod(t, 60)
            log.info(f"All done: {int(m):02d}min{s:.2f}s")
        else:
            if t < 3600 * 24:
                m, s = divmod(t, 60)
                h, m = divmod(m, 60)
                log.info(f"All done: {int(h):d}h{int(m):02d}min{s:.2f}s")
            else:
                if t < 3600 * 24 * 365:
                    m, s = divmod(t, 60)
                    h, m = divmod(m, 60)
                    d, h = divmod(h, 24)
                    log.info(
                        f"All done: {int(d):d}d{int(h):02d}h{int(m):02d}min{s:.2f}s"
                    )


if __name__ == "__main__":
    main()
