"""Script to run MaxiTrack inference."""
#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import argparse
import logging as log
import math
import os
import random as rd
import sys
import time

import numpy as np
import tqdm
from astropy.io import fits

from maximask_and_maxitrack import utils


class MaxiTrack_inference(object):
    """Class to run MaxiTrack inference."""

    im_size = 400

    def __init__(self, im_path, net_dir, prior, frac, batch_size):
        """Initializes the MaxiTrack_inference class.
        Args:
            im_path (string): path to the images to be processed. This can be a fits file, a directory or a list file.
            net_dir (string): path to the MaxiTrack neural network directory.
            prior (float32): tracking error prior. None if the prior modification is not requested.
            frac (float32): value specifying a fraction of all the HDUs to use to speed up processing.
            batch_size (int): batch size to use for inference.
        """

        self.im_path = im_path
        self.net_dir = net_dir
        self.prior = prior
        self.frac = frac
        self.batch_size = batch_size

    @utils.timeit
    def process_all(self):
        """Processes all the requested images.
        This will write the predictions in the "maxitrack.out" file.
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

            # process each file of file list
            for file_name in tqdm.tqdm(file_list, desc="ALL FILES"):
                log.info(f"Starting {file_name}")
                self.process_file(file_name, tf_model)
                log.info(f"{file_name} done")
        else:
            log.error(f"No file to process given input path <{self.im_path}>")

    def process_file(self, file_name, tf_model):
        """Processes a given fits file.
        This will append the predictions in the "maxitrack.out" file.

        Args:
            file_name (string): name of the file to process.
            tf_model (tf.keras.Model): MaxiTrack tensorflow model.
        """

        all_preds = []

        # make hdu tasks
        hdu_task_list = self.make_hdu_tasks(file_name)

        # if at least one usable HDU
        if len(hdu_task_list):
            # apply eventual fraction option
            if self.frac != 1:
                nb_hdus = len(hdu_task_list)
                nb_hdus_to_use = int(self.frac * nb_hdus)
                if nb_hdus_to_use == 0:
                    nb_hdus_to_use = 1
                log.info(
                    f"Using fraction option with {self.frac} fraction of {nb_hdus} HDUs"
                )
                log.info(f"Selecting {nb_hdus_to_use} random HDUs")
                hdu_task_list = rd.sample(hdu_task_list, nb_hdus_to_use)
            else:
                log.info("Using all available HDUs")

            # go through all HDUs
            for hdu_idx, hdu_type, hdu_shape in hdu_task_list:
                log.info(f"HDU {hdu_idx}")

                # get raw predictions
                preds, t = self.process_hdu(file_name, hdu_idx, tf_model)
                log.info(
                    f"Whole processing time (incl. preprocessing): {t:.2f}s, {np.prod(hdu_shape)/(t*1e06):.2f}MPix/s"
                )

                # append the results
                for pred in preds:
                    all_preds.append(pred)

            final_res = np.mean(all_preds)

            # write file
            with open("maxitrack.out", "a") as fd:
                fd.write(f"{file_name} {final_res:.4f}\n")
        else:
            log.info(f"Skipping {file_name} because no HDU was found to be processed")

    def make_hdu_tasks(self, file_name):
        """Makes the hdu task list for the given file to process.

        Args:
            file_name (string): name of the file to process.
        Returns:
            hdu_task_list (list): list of hdu tasks for the file to process.
                                  each element contains the hdu number and the type and shape of the data contained in the HDU.
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
                    hdu_task_list.append([spec_hdu_idx, hdu_type, hdu_shape])
                else:
                    log.info(
                        f"Ignoring HDU {spec_hdu_idx} because not adequate data format"
                    )
        else:
            with fits.open(file_name) as file_hdu:
                nb_hdu = len(file_hdu)
                for k in range(nb_hdu):
                    check, hdu_type = utils.check_hdu(file_hdu[k], self.im_size)
                    if check:
                        hdu_shape = file_hdu[k].data.shape
                        hdu_task_list.append([k, hdu_type, hdu_shape])
                    else:
                        log.info(f"Ignoring HDU {k} because not adequate data format")

        return hdu_task_list

    @utils.timeit
    def process_hdu(self, file_name, hdu_idx, tf_model):
        """Processes the hdu of a given file according to the task.

        Args:
            file_name (string): name of the file to process.
            hdu_idx (int): index of the HDU to process.
            tf_model (tf.keras.Model): MaxiTrack tensorflow model.
        Returns:
            out_array (np.ndarray): MaxiTrack predictions over the image.
        """

        # make file name
        _, im_path_ext = os.path.splitext(file_name)
        if "[" in im_path_ext:
            file_name = file_name.split("[")[0]

        # get input data
        with fits.open(file_name) as file_hdu:
            hdu = file_hdu[hdu_idx]
            im_data = hdu.data

        # get list of block coordinate to process
        h, w = im_data.shape
        block_coord_list = self.get_block_coords(h, w)

        # preprocessing
        log.info("Preprocessing...")
        im_data, t = utils.image_norm(im_data)
        log.info(f"Preprocessing done in {t:.2f}s, {h*w/(t*1e06):.2f}MPix/s")

        # process all the blocks by batches
        nb_blocks = len(block_coord_list)
        out_array = np.zeros([nb_blocks], dtype=np.float32)
        if nb_blocks <= self.batch_size:
            # only one (possibly not full) batch to process
            res = self.process_batch(im_data, tf_model, block_coord_list)
            out_array = res.numpy()
        else:
            # several batches to process + one last possibly not full
            nb_batch = nb_blocks // self.batch_size
            for b in tqdm.tqdm(range(nb_batch), desc="INFERENCE"):
                batch_coord_list = block_coord_list[
                    b * self.batch_size : (b + 1) * self.batch_size
                ]
                res = self.process_batch(im_data, tf_model, batch_coord_list)
                out_array[b * self.batch_size : (b + 1) * self.batch_size] = res
            rest = nb_blocks - nb_batch * self.batch_size
            if rest:
                batch_coord_list = block_coord_list[-rest:]
                res = self.process_batch(im_data, tf_model, batch_coord_list)
                out_array[b * self.batch_size : b * self.batch_size + rest] = res

        return out_array

    def get_block_coords(self, h, w):
        """Gets the coordinate list of blocks to process.

        Args:
            h (int): full height of the image to process.
            w (int): full width of the image to process.
        Returns:
            coord_list (list): list of coordinates of blocks to process.
        """

        coord_list = []
        h_r, w_r = h % self.im_size, w % self.im_size
        for y in range(h_r, h - self.im_size + 1, self.im_size):
            for x in range(w_r, w - self.im_size + 1, self.im_size):
                coord_list.append([x, y])

        return coord_list

    def process_batch(self, im_data, tf_model, batch_coord_list):
        """Process a batch of inputs.

        Args:
            im_data (np.ndarray): image data to process.
            tf_model (tf.keras.Model): MaxiTrack tensorflow model.
            batch_coord_list (list): list of coordinates of block to process for this batch.
        Returns:
            res (tf.Tensor): MaxiTrack predictions for each image of the batch.
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
        res = tf_model(inp, False)[:, 1]

        # apply eventual prior modification
        if self.prior is not None:
            prior_factor = (1 - self.prior) / self.prior
            res = res / (res + prior_factor * (1 - res))

        return res


def main():

    ### parameter parser
    parser = argparse.ArgumentParser(description="MaxiTrack command line parameters:")

    # positional parameter
    parser.add_argument(
        "im_path", type=str, help="path to the image(s) to be processed"
    )

    # optional parameters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    net_dir = os.path.join(script_dir, "model")
    parser.add_argument(
        "--net_dir",
        type=str,
        help="neural network graphs and weights directory. Default is </path_to_root/maximask_and_maxitrack/maxitrack/model>",
        default=net_dir,
    )
    parser.add_argument(
        "--prior", type=float, help="prior value to use. Default is 0.05", default=0.05
    )
    parser.add_argument(
        "--frac",
        type=float,
        help="value specifying a fraction of all the HDUs to use to speed up processing. Default is 1",
        default=1,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="neural network batch size. Default is 16. You might want to use a lower value if you have RAM issues",
        default=16,
    )
    parser.add_argument(
        "-v", "--verbose", help="activate output verbosity", action="store_true"
    )

    # retrieve parameters
    args = parser.parse_args()
    im_path = args.im_path
    net_dir = args.net_dir
    prior = args.prior
    frac = args.frac
    if frac <= 0 or frac > 1:
        raise ValueError(
            f"Provided fraction parameter is {frac} but should be between 0 and 1"
        )
    batch_size = args.batch_size
    verbose = args.verbose
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")

    # build MaxiTrack_inference object
    mt_inf = MaxiTrack_inference(im_path, net_dir, prior, frac, batch_size)

    # process all
    _, t = mt_inf.process_all()
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
