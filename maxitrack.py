#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import os
import sys
import math
import time
import argparse
import numpy as np
import random as rd
from astropy.io import fits

import utils

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 


def process_file(sess, src_im_s):
    """ Process one fits file: several behaviours depending on <IM_PATH> and <src_im_s>
    """

    if src_im_s!="" and IM_PATH[-5:]!=".list":
        # processing images of a given directory
        im_path = IM_PATH + "/" + src_im_s
    elif IM_PATH[-5:]==".list":
        # processing images of a given list file
        im_path = src_im_s
    else:
        # processing a single image (hdu)
        im_path = IM_PATH

    if im_path[-1]=="]":
        # process the specified hdu
        spec_hdu = im_path.split("[")[1].split("]")[0]
        n = int(len(spec_hdu)+2)
        with fits.open(im_path[:-n]) as src_im_hdu:
            if int(spec_hdu)>len(src_im_hdu):
                print("Error: requesting hdu " + spec_hdu + " when image has only " + str(len(src_im_hdu)) + " hdu(s)")
                print("Exiting...")
                sys.exit()
            src_im = src_im_hdu[int(spec_hdu)].data

        if len(src_im_hdu[int(spec_hdu)].shape)==2 and type(src_im[0,0]) in [np.float32, np.float16, np.int32, np.int16, np.uint16, np.float64]:
            src_im = src_im.astype(np.float32)
            h,w = src_im.shape

            if np.any(src_im):
                # dynamic compression
                t1 = utils.dynamic_compression(src_im)
                if VERB: 
                    speed1 = str(round((h*w)/(t1*1000000), 3))
                    print(IM_PATH + " dynamic compression done in " + str(t1) + " s: " + speed1 + " MPix/s")

                # inference
                results = []
                t2 = process_hdu(src_im, results, sess)
                if VERB: 
                    speed2 = str(round((h*w)/(t2*1000000), 3))
                    print(IM_PATH + " inference done in " + str(t2) + " s: " + speed2 + " MPix/s")
            else:
                # full zero image
                results = [0]
                if VERB: print(IM_PATH + " inference done (image is null, output is null)")
                
            if VERB:
                if results!=[0]:
                    speedhdu = str(round((h*w)/((t1+t2)*1000000), 3))
                    print(IM_PATH + " done in " + str(t1+t2) + " s: " + speedhdu + " MPix/s")

            # writing
            with open("maxitrack.out", "a") as fd:
                fd.write(IM_PATH + " " + str(sum(results)/float(len(results))) + "\n")

        else:
            print("Error: requested hdu " + spec_hdu + " does not contain either 2D data or float data")
            print("Exiting...")
            sys.exit()
    else:
        timelog = []
        all_results = []
        # process all hdus containing data or some random hdus according to frac
        with fits.open(im_path) as src_im_hdu:
            nb_hdu = len(src_im_hdu)
            if FRAC==-1:
                hdus = np.arange(nb_hdu)
            elif FRAC>0 and FRAC<nb_hdu:
                hdus = rd.sample(np.arange(nb_hdu), FRAC)
                if VERB: print("Using HDUs " + str(hdus) + " among the " + str(nb_hdu))
            else:
                if VERB: print("Requesting to use " + str(FRAC) + " HDUs with --frac option when image has " + str(nb_hdu) + " HDUs...\nMaxiTrack adopts default behaviour and will use all HDUs")
                hdus = np.arange(nb_hdu)

            for k in hdus:
                results = []
                src_im = src_im_hdu[k].data

                if len(src_im_hdu[k].shape)==2 and type(src_im[0,0]) in [np.float32, np.float16, np.int32, np.int16, np.uint16, np.float64]:
                    src_im = src_im.astype(np.float32)
                    h,w = src_im.shape
                    
                    if np.any(src_im):
                        # dynamic compression
                        t1 = utils.dynamic_compression(src_im)
                        if VERB: 
                            speed1 = str(round((h*w)/(t1*1000000), 3))
                            print("HDU " + str(k) + "/" + str(nb_hdu-1) + " dynamic compression done in " + str(t1) + " s: " + speed1 + " MPix/s")

                        # inference
                        t2 = process_hdu(src_im, results, sess)
                        if VERB:
                            speed2 = str(round((h*w)/(t2*1000000), 3))
                            print("HDU " + str(k) + "/" + str(nb_hdu-1) + " inference done in " + str(t2) + " s: " + speed2 + " MPix/s")
                            speedhdu = str(round((h*w)/((t1+t2)*1000000), 3))
                        timelog.append(t1+t2)
                    else:
                        # full zero image
                        if VERB: print("HDU " + str(k) + "/" + str(nb_hdu-1) + " inference done (image is null, thus ignored)")
                    
                else:
                    # if this seems not to be data then ignore
                    if VERB: print("HDU " + str(k) + "/" + str(nb_hdu-1) + " done (just ignored as it is not 2D float data)") 

                if len(results):
                    all_results.append(sum(results)/float(len(results)))

            # writing
            with open("maxitrack.out", "a") as fd:
                if len(all_results):
                    fd.write(im_path + " " + str(sum(all_results)/float(len(all_results))) + "\n")
                else:
                    fd.write(im_path + " " + str(None) + "\n")


@utils.timeit
def process_hdu(src_im, results, sess):
    """ Process one hdu: cut it into batches and process each batch
    Compute prior modification and/or thresholding is requested
    """

    h,w = src_im.shape
    if h<IM_SIZE or w<IM_SIZE:
        print("One of the two image dimension is less than " + str(IM_SIZE) + " : not supported yet")
        print("Exiting...")
        sys.exit()
        
    # managing borders
    h_r, w_r = h%IM_SIZE, w%IM_SIZE

    # list of positions to make inference on
    tot_l = []
    for y in range(h_r//2, h-IM_SIZE-h_r//2+1, IM_SIZE):
        for x in range(w_r//2, w-IM_SIZE-w_r//2+1, IM_SIZE):
            tot_l.append([x, y])
    
    # if less inferences than batch size do it in one pass
    if len(tot_l)<=BATCH_S:
        process_batch(src_im, results, sess, len(tot_l), tot_l, 0, len(tot_l))
    # otherwise iterate over all batches to do
    else:
        nb_step = len(tot_l)//BATCH_S+1
        for st in range(nb_step-1):
            if st<nb_step-1:
                process_batch(src_im, results, sess, BATCH_S, tot_l, st*BATCH_S, (st+1)*BATCH_S)
        # manage the last (incomplete) batch
        re = len(tot_l)-(nb_step-1)*BATCH_S
        if re:
            process_batch(src_im, results, sess, re, tot_l, (nb_step-1)*BATCH_S, len(tot_l))


def process_batch(src_im, results, sess, batch_s, tot_l, first_p, last_p):
    """ Process one batch of subimage: get corresponding predictions depending on subimage position in the field
    """
    
    h,w = src_im.shape

    inp = np.zeros([batch_s, IM_SIZE, IM_SIZE], dtype=np.float32)
    
    # prepare inputs and make inference
    k = 0
    for coord in tot_l[first_p:last_p]:
        inp[k] = src_im[coord[1]:coord[1]+IM_SIZE, coord[0]:coord[0]+IM_SIZE]
        k += 1
    tmp_res = sess.run("predictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "batch_s:0": k, "drop:0": 1.0})[:,0]
    prior_factor = (1-PRIOR)/PRIOR
    tmp_res = tmp_res/(tmp_res + prior_factor*(1-tmp_res))
    results += list(tmp_res)


def setup_params():
    """ Read all parameters from command line and from parameter files
    """

    # parameter parser
    parser = argparse.ArgumentParser(description='MaxiTrack command line parameters:')

    # necessary parameters
    parser.add_argument("im_path", type=str, help='path the image(s) to be processed')

    # optional parameters
    parser.add_argument("--net_path", type=str, help='path to the neural network graphs and weights directory. Default is </abs_path_to_rep/model>', default=os.path.dirname(os.path.abspath(__file__)) + "/track_model")
    parser.add_argument("--prior_value", type=float, help='float defining the expected prior in data. Default is 0.05', default=0.05)
    parser.add_argument("--frac", type=int, help='int defining the number of HDU to use. Default is -1, meaning that MaxiTrack will use all HDU', default=-1)
    parser.add_argument("--batch_size", type=int, help='neural network batch size. Default is 8. You might want to use a lower value if you have RAM issues', default=8)
    parser.add_argument("-v", "--verbose", help="activate output verbosity", action="store_true")

    # read arguments
    args = parser.parse_args()

    global IM_PATH

    global NET_PATH
    global PRIOR
    global FRAC
    global BATCH_S
    global VERB

    IM_PATH = args.im_path

    NET_PATH = args.net_path
    PRIOR = args.prior_value
    FRAC = args.frac
    BATCH_S = args.batch_size
    VERB = args.verbose
    

def main():
    """ Main function
    """

    # setup all parameters
    setup_params()
    
    config = tf.ConfigProto()
    if tf.test.is_gpu_available():
        config.gpu_options.allow_growth = True
        print("MaxiTrack is using GPU")
    else:
        print("MaxiTrack is using CPU")
        
    # open tf session first so all is done in one single session
    with tf.Session(config=config) as sess:
        nsaver = tf.train.import_meta_graph(NET_PATH + "/model.meta")
        nsaver.restore(sess, NET_PATH + "/model")
        
        if os.path.isfile(IM_PATH) or IM_PATH[-1]=="]":
            if IM_PATH[-5:]==".list":
                # process all images of list file
                with open(IM_PATH) as fd:
                    lines = fd.readlines()
                for src_im_s in lines:
                    if "fits" in src_im_s:
                        src_im_s = src_im_s.rstrip()
                        if VERB: print("Processing " + src_im_s + " from " + IM_PATH + " list file")
                        process_file(sess, src_im_s)
                        if VERB: print
            elif "fits" in IM_PATH:
                # process the single file image
                if VERB: print("Processing " + IM_PATH)
                process_file(sess, "")
                if VERB: print
        else:
            # process all the images of the directory
            if VERB: print("Processing " + IM_PATH)
            for src_im_s in os.listdir(IM_PATH):
                if "fits" in src_im_s:
                    if VERB: print("Processing " + IM_PATH + "/" + src_im_s)
                    process_file(sess, src_im_s)
                    if VERB: print


if __name__=="__main__":
    # parameter values that should never change and should not be changed by user
    IM_SIZE = 400
    IM2 = IM_SIZE//2
    IM4 = IM_SIZE//4

    # parameter which values are read from command line
    IM_PATH = None

    NET_PATH = None
    PRIOR = 0.05
    FRAC = None
    BATCH_S = None
    VERB = None

    main()
