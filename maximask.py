#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import os
import sys
import math
import time
import argparse
import numpy as np
import scipy.interpolate as interp
import scipy.signal as sign
from astropy.io import fits

import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

def timeit(f):
    """ A decorator to measure execution time of a function
    """
    
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()
        
        return round(te-ts, 3)

    return timed


def background_est(im):
    """ Process a Sextractor like background estimation
    """
    
    h,w = im.shape
    mesh_size = 200

    # get the number blocks to process
    h_blocks = (h-h%mesh_size)//mesh_size
    w_blocks = (w-w%mesh_size)//mesh_size
    x_l, y_l = [], []
    for y in range(h_blocks):
        for x in range(w_blocks):
            x_l.append(x*mesh_size)
            y_l.append(y*mesh_size)
    
    # process the blocks
    nb_blocks = len(x_l)
    z = [None]*nb_blocks
    for b in range(nb_blocks):
        x_b, x_e = x_l[b], min(x_l[b]+mesh_size, w)
        y_b, y_e = y_l[b], min(y_l[b]+mesh_size, h)

        mesh = im[y_b:y_e, x_b:x_e]
            
        tmp_mask = np.ones_like(mesh)
        idx = np.where(tmp_mask)
        init_std = np.std(mesh[idx[0], idx[1]])
        tmp_std = init_std
        tmp_med = np.median(mesh[idx[0], idx[1]])
        k = 3.0
        while True:
            to_keep = np.logical_and(tmp_mask*(mesh >= tmp_med-k*tmp_std), tmp_mask*(mesh <= tmp_med+k*tmp_std))
            to_keep = np.logical_and(to_keep, np.logical_not(mesh<=0))
            if np.all(to_keep)==np.all(tmp_mask):
                break
            else:
                tmp_mask *= to_keep
                idx = np.where(tmp_mask)
                tmp_std = np.std(mesh[idx[0], idx[1]])
                tmp_med = np.median(mesh[idx[0], idx[1]])
        idx = np.where(tmp_mask)
        if tmp_std > init_std - 0.01*init_std or tmp_std < init_std + 0.01*init_std:
            b_v = np.mean(mesh[idx[0], idx[1]])
        else:
            b_v = 2.5*np.median(mesh[idx[0], idx[1]]) - 1.5*np.mean(mesh[idx[0], idx[1]])
        z[b] = b_v

    # build the little mesh to median filter and to interpolate
    to_interp = np.zeros([h_blocks, w_blocks])
    for b in range(nb_blocks):
        to_interp[int(y_l[b]/mesh_size), int(x_l[b]/mesh_size)] = z[b]
        
    to_interp2 = np.zeros([h_blocks+2, w_blocks+2])
    to_interp2[1:-1, 1:-1] = to_interp

    to_interp2[0,1:-1] = to_interp[0,:]
    to_interp2[1:-1,0] = to_interp[:,0]
    to_interp2[-1,1:-1] = to_interp[-1,:]
    to_interp2[1:-1,-1] = to_interp[:,-1]

    # median filter
    to_interp2 = sign.medfilt(to_interp2, 3)
    
    # replace the 0 in the corners to avoid corner artefacts
    to_interp2[0,0] = to_interp2[1,1]
    to_interp2[0,-1] = to_interp2[1,-2]
    to_interp2[-1,0] = to_interp2[-2,1]
    to_interp2[-1,-1] = to_interp2[-2,-2]

    # interpolate across the blocks
    f = interp.RectBivariateSpline(np.arange(-mesh_size/2, (h_blocks+1)*mesh_size, mesh_size), np.arange(-mesh_size/2, (w_blocks+1)*mesh_size, mesh_size), to_interp2)
    back_val = f(np.arange(h), np.arange(w)).astype(np.float32)

    return back_val 


@timeit
def dynamic_compression(im):
    """ Dynamical compressing: sky background subtraction and sigma normalizing
    """

    # just a check to remove eventual nan values
    np.place(im, im==np.isnan, 0)

    # dynamic compression
    bg_map = background_est(im)
    im -= bg_map
    sig = np.std(im)
    im /= sig


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

        if len(src_im_hdu[int(spec_hdu)].shape)==2 and type(src_im[0,0]) in [np.float32, np.float16, np.int32, np.int16, np.uint16]:
            src_im = src_im.astype(np.float32)
            h,w = src_im.shape

            if PROBA_THRESH:
                if SINGLE_MASK:
                    results = np.zeros([1, h, w], dtype=np.uint16)
                else:
                    results = np.zeros([NB_CL, h, w], dtype=np.uint8)
            else:
                results = np.zeros([NB_CL, h, w], dtype=np.float32)

            if np.any(src_im):
                full_zero = False
                # dynamic compression
                t1 = dynamic_compression(src_im)
                if VERB: 
                    speed1 = str(round((h*w)/(t1*1000000), 3))
                    print(IM_PATH + " dynamic compression done in " + str(t1) + " s: " + speed1 + " MPix/s")

                # inference
                t2 = process_hdu(src_im, results, sess)
                if VERB: 
                    speed2 = str(round((h*w)/(t2*1000000), 3))
                    print(IM_PATH + " inference done in " + str(t2) + " s: " + speed2 + " MPix/s")
                    speedhdu = str(round((h*w)/((t1+t2)*1000000), 3))
                    print(IM_PATH + " done in " + str(t1+t2) + " s: " + speedhdu + " MPix/s")
            else:
                # full zero image
                full_zero = True
                if VERB: print(IM_PATH + " inference done (image is null, output is null)")
                
            # writing
            hdu = fits.PrimaryHDU(results)
            fill_hdu_header(hdu)
            tw = write_hdu(hdu, im_path[:-n].split(".fits")[0] + ".masks" + spec_hdu + ".fits")
            if VERB: 
                if not full_zero:
                    print(im_path[:-n].split(".fits")[0] + ".masks" + spec_hdu + ".fits written to disk in " + str(tw) + " s")
                    speedt = str(round((h*w)/((t1+t2+tw)*1000000), 3))
                    print("Total time: " + str(t1+t2+tw) + " s: " + speedt + " MPix/s")
        else:
            print("Error: requested hdu " + spec_hdu + " does not contain 2D data or supported type")
            print("Exiting...")
            sys.exit()
    else:
        timelog = []
        # process all hdus containing data
        with fits.open(im_path) as src_im_hdu:
            nb_hdu = len(src_im_hdu)
            hdu = fits.HDUList()
            for k in range(nb_hdu):
                src_im = src_im_hdu[k].data

                if len(src_im_hdu[k].shape)==2 and type(src_im[0,0]) in [np.float32, np.float16, np.int32, np.int16, np.uint16]:
                    src_im = src_im.astype(np.float32)
                    h,w = src_im.shape

                    if PROBA_THRESH:
                        if SINGLE_MASK:
                            results = np.zeros([1, h, w], dtype=np.uint16)
                        else:
                            results = np.zeros([NB_CL, h, w], dtype=np.uint8)
                    else:
                        results = np.zeros([NB_CL, h, w], dtype=np.float32)
                    
                    if np.any(src_im):
                        # dynamic compression
                        t1 = dynamic_compression(src_im)
                        if VERB: 
                            speed1 = str(round((h*w)/(t1*1000000), 3))
                            print("HDU " + str(k) + "/" + str(nb_hdu-1) + " dynamic compression done in " + str(t1) + " s: " + speed1 + " MPix/s")

                        # inference
                        t2 = process_hdu(src_im, results, sess)
                        if VERB: 
                            speed2 = str(round((h*w)/(t2*1000000), 3))
                            print("HDU " + str(k) + "/" + str(nb_hdu-1) + " inference done in " + str(t2) + " s: " + speed2 + " MPix/s")
                            speedhdu = str(round((h*w)/((t1+t2)*1000000), 3))
                        full_zero = False
                        timelog.append(t1+t2)
                        if VERB: print("HDU " + str(k) + "/" + str(nb_hdu-1) + " done in " + str(t1+t2) + " s: " + speedhdu + " MPix/s")
                    else:
                        # full zero image
                        if VERB: print("HDU " + str(k) + "/" + str(nb_hdu-1) + " inference done (image is null, output is null)")
                        full_zero = True
    
                    if k==0:
                        m_hdu = fits.PrimaryHDU(results)
                        fill_hdu_header(m_hdu)
                        hdu.append(m_hdu)
                    else:
                        sub_hdu = fits.ImageHDU(results)
                        fill_hdu_header(sub_hdu)
                        hdu.append(sub_hdu)
                else:
                    # if this seems not to be data then copy the hdu
                    hdu.append(src_im_hdu[k])
                    if VERB: print("HDU " + str(k) + "/" + str(nb_hdu-1) + " done (just copied as it is not 2D data or supported type)") 
           
            tw = write_hdu(hdu, im_path.split(".fits")[0] + ".masks.fits")
            if VERB: 
                print(im_path.split(".fits")[0] + ".masks.fits written to disk in " + str(tw) + " s")
                if len(timelog):
                    tt = sum(timelog) + tw
                    speedt = str(round((h*w)*len(timelog)/(tt*1000000), 3))
                    print("Total time: " + str(tt) + " s: " + speedt + " MPix/s")


@timeit
def process_hdu(src_im, results, sess):
    """ Process one hdu: cut it into batches and process each batch
    Compute prior modification and/or thresholding is requested
    """

    h,w = src_im.shape
    if h<IM_SIZE or w<IM_SIZE:
        print("One of the two image dimension is less than " + str(IM_SIZE) + " : not supported yet")
        print("Exiting...")
        sys.exit()

    # list of positions to make inference on
    tot_l = []
    for y in range(0, h-IM_SIZE+1, IM2):
        for x in range(0, w-IM_SIZE+1, IM2):
            tot_l.append([x, y])
    if h%IM_SIZE:
        for x in range(0, w-IM_SIZE+1, IM2):
            tot_l.append([x, h-IM_SIZE])
    if w%IM_SIZE:
        for y in range(0, h-IM_SIZE+1, IM2):
            tot_l.append([w-IM_SIZE, y])
    if w%IM_SIZE and h%IM_SIZE: 
        tot_l.append([w-IM_SIZE, h-IM_SIZE])

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
        
    if PRIOR_MODIF and not PROBA_THRESH:
        tmp_results = sess.run("ppredictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "flags:0": CLASSES, "priors:0": PRIOR_F})
    elif not PRIOR_MODIF and PROBA_THRESH and not SINGLE_MASK:
        tmp_results = sess.run("tpredictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "flags:0": CLASSES, "priors:0": PRIOR_F, "thresh:0": THRESH})
    elif PRIOR_MODIF and PROBA_THRESH and not SINGLE_MASK:
        tmp_results = sess.run("tpredictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "flags:0": CLASSES, "priors:0": PRIOR_F, "thresh:0": THRESH})
    elif not PRIOR_MODIF and not PROBA_THRESH:
        tmp_results = sess.run("fpredictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "flags:0": CLASSES})
    elif SINGLE_MASK:
        tmp_results = sess.run("spredictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1]), "flags:0": CLASSES, "priors:0": PRIOR_F, "thresh:0": THRESH, "2powers:0": POWERS})

    # copy in final mask
    k = 0
    for x,y in tot_l[first_p:last_p]:
        results[:, IM4+y:y+IM_SIZE-IM4, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][:, IM4:IM_SIZE-IM4, IM4:IM_SIZE-IM4]
        if x==0:
            results[:, IM4+y:y+IM_SIZE-IM4, x:x+IM4] = tmp_results[k][:, IM4:IM_SIZE-IM4, :IM4]
        if y==0:
            results[:, y:y+IM4, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][:, :IM4, IM4:IM_SIZE-IM4]
        if x==w-IM_SIZE:
            results[:, IM4+y:y+IM_SIZE-IM4, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][:, IM4:IM_SIZE-IM4, IM_SIZE-IM4:IM_SIZE]
        if y==h-IM_SIZE:
            results[:, y+IM_SIZE-IM4:y+IM_SIZE, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][:, IM_SIZE-IM4:IM_SIZE, IM4:IM_SIZE-IM4]
        if x==0 and y==0:
            results[:, y:y+IM4, x:x+IM4] = tmp_results[k][:, :IM4, :IM4]
        if x==0 and y==h-IM_SIZE:
            results[:, y+IM_SIZE-IM4:y+IM_SIZE, x:x+IM4] = tmp_results[k][:, IM_SIZE-IM4:IM_SIZE, :IM4]
        if x==w-IM_SIZE and y==0:
            results[:, y:y+IM4, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][:, :IM4, IM_SIZE-IM4:IM_SIZE]
        if x==w-IM_SIZE and y==h-IM_SIZE:
            results[:, y+IM_SIZE-IM4:y+IM_SIZE, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][:, IM_SIZE-IM4:IM_SIZE, IM_SIZE-IM4:IM_SIZE]
        k += 1


def fill_hdu_header(hdu):
    """ Fill the hdu header with corresponding informations
    """

    hdu.header['MM_UTC'] = time.asctime(time.gmtime())
    hdu.header.comments['MM_UTC'] = "MaxiMask UTC processing date"
    hdu.header['MM_LOC'] = time.asctime(time.localtime())
    hdu.header.comments['MM_LOC'] = "MaxiMask LOC processing date"
    hdu.header['MM_TF_V'] = tf.__version__
    hdu.header.comments['MM_TF_V'] = "MaxiMask tensorflow version used"
    hdu.header['MM_HBACK'] = H_BACK.upper()
    hdu.header.comments['MM_HBACK'] = "MaxiMask tensorflow hardware used"

    if not PRIOR_MODIF:
        hdu.header['PRIOR'] = "No prior modification"
    if not PROBA_THRESH:
        hdu.header['THRESH'] = "No probability thresholding"

    if PRIOR_MODIF or PROBA_THRESH:
        for cl in range(NB_CL):
            if SINGLE_MASK:
                hdu.header[CLASS_ABBRV[cl]] = POWERS[cl]
                hdu.header.comments[CLASS_ABBRV[cl]] = CLASS_NAMES[cl] + " mask value"

                if PRIOR_MODIF:
                    hdu.header[CLASS_ABBRV[cl] + '_PR'] = str(round(PRIORS[cl], 6))
                    hdu.header.comments[CLASS_ABBRV[cl] + '_PR'] = CLASS_NAMES[cl] + " class prior"
                    
                hdu.header[CLASS_ABBRV[cl] + '_TH'] = str(round(THRESH[cl], 3))
                hdu.header.comments[CLASS_ABBRV[cl] + '_TH'] = CLASS_NAMES[cl] + " class threshold"
            else:
                hdu.header['M' + str(cl)] = CLASS_NAMES[cl]
                hdu.header.comments['M' + str(cl)] = "Mask " + str(cl) + " class name"
                if PRIOR_MODIF:
                    hdu.header['M' + str(cl) + '_PR'] = str(round(PRIORS[cl], 6))
                    hdu.header.comments['M' + str(cl) + '_PR'] = "Mask " + str(cl) + " class prior"
                if PROBA_THRESH:
                    hdu.header['M' + str(cl) + '_TH'] = str(round(THRESH[cl], 3))
                    hdu.header.comments['M' + str(cl) + '_TH'] = "Mask " + str(cl) + " class threshold"


@timeit
def write_hdu(hdu, name):
    """ Write a fits image to disk (separated in a function just to be decorated)
    """
   
    hdu.writeto(name, overwrite=True)


def str2bool(v):
    """ Translating possible boolean inputs to boolean type
    """

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_params():
    """ Read all parameters from command line and from parameter files
    """

    # parameter parser
    parser = argparse.ArgumentParser(description='MaxiMask command line parameters:')

    # positional parameter
    parser.add_argument("im_path", type=str, help='path the image(s) to be processed')

    # optional parameters
    parser.add_argument("--net_path", type=str, help='path to the neural network graphs and weights directory. Default is </abs_path_to_rep/model>', default=os.path.dirname(os.path.abspath(__file__)) + "/model")
    parser.add_argument("--prior_modif", type=str2bool, help='bool indicating if probability maps should be prior modified. Default is True', default=True)
    parser.add_argument("--proba_thresh", type=str2bool, help='bool indicating if probability maps should be thresholded. Default is True', default=True)
    parser.add_argument("--single_mask", type=str2bool, help='bool indicating if resulting masks are joined in a single mask using powers of two', default=False)
    parser.add_argument("--batch_size", type=int, help='neural network batch size. Default is 8. You might want to use a lower value if you have RAM issues', default=8)
    parser.add_argument("-v", "--verbose", help="activate output verbosity", action="store_true")

    # read arguments with parser
    args = parser.parse_args()

    global IM_PATH

    global NET_PATH
    global PRIOR_MODIF
    global PROBA_THRESH
    global SINGLE_MASK
    global BATCH_S
    global VERB

    IM_PATH = args.im_path

    NET_PATH = args.net_path
    PRIOR_MODIF = args.prior_modif
    PROBA_THRESH = args.proba_thresh
    SINGLE_MASK = args.single_mask
    if SINGLE_MASK and not PROBA_THRESH:
        print("Error: contradictory parameters: requesting a single mask but not requesting thresholding")
        print("Exiting...")
        sys.exit()
    BATCH_S = args.batch_size
    VERB = args.verbose

    # read classes.flags
    global CLASSES
    global NB_CL
    global TR_PRIORS
    try:
        with open("classes.flags") as fd:
            lines = fd.readlines()
            CLASSES = []
            if len(lines)!=TNB_CL:
                print("Error: classes.flags must contain " + str(TNB_CL) + " lines, one for each class")
                print("Exiting...")
                sys.exit()
            for k in range(TNB_CL):
                try:
                    if int(lines[k].split()[1]):
                        CLASSES.append(k)
                except (ValueError, IndexError) as e:
                    print("Error: classes.flags file exists but may not be well formatted (at least one class does not have a proper boolean flag)")
                    print("Exiting...")
                    sys.exit()
        NB_CL = len(CLASSES)
        tmp = np.zeros([NB_CL], dtype=np.float32)
        for k in range(NB_CL):
            tmp[k] = TR_PRIORS[CLASSES[k]]
        TR_PRIORS = tmp
    except IOError:
        if VERB:
            print("No classes.flags found: MaxiMask will output probabilities/masks for all classes")
    
    # if prior modif is requested read classes.priors
    global PRIORS
    global PRIOR_F
    if PRIOR_MODIF:
        try:
            with open("classes.priors") as fd:
                lines = fd.readlines()
                PRIORS = np.zeros([NB_CL], dtype=np.float32)
                if len(lines)!=TNB_CL:
                    print("Error: classes.priors must contain " + str(TNB_CL) + " lines, one for each class")
                    print("Exiting...")
                    sys.exit()
                for k in range(NB_CL):
                    try:
                        PRIORS[k] = float(lines[CLASSES[k]].split()[1])
                    except (ValueError, IndexError) as e:
                        print("Error: classes.priors file exists but may not be well formatted (at least one requested class does not have a proper prior)")
                        print("Exiting...")
                        sys.exit()
            PRIOR_F = (TR_PRIORS/(1-TR_PRIORS))*((1-PRIORS)/PRIORS)
        except IOError:
            tmp = np.zeros([NB_CL], dtype=np.float32)
            for k in range(NB_CL):
                tmp[k] = PRIORS[CLASSES[k]]
            PRIORS = tmp
            PRIOR_F = (TR_PRIORS/(1-TR_PRIORS))*((1-PRIORS)/PRIORS)
            if VERB:
                print("No classes.priors file found whereas prior modification is requested: using default priors")
    
    # if probability thresholding is requested read classes.thresh
    global THRESH
    if PROBA_THRESH:
        try:
            with open("classes.thresh") as fd:
                lines = fd.readlines()
                THRESH = np.zeros([NB_CL], dtype=np.float32)
                if len(lines)!=TNB_CL:
                    print("Error: classes.thresh must contain " + str(TNB_CL) + " lines, one for each class")
                    print("Exiting...")
                    sys.exit()
                for k in range(NB_CL):
                    try:
                        THRESH[k] = float(lines[CLASSES[k]].split()[1])
                    except (ValueError, IndexError) as e:
                        print("Error: classes.thresh file exists but may not be well formatted (at least one requested class does not have a proper threshold)")
                        print("Exiting...")
                        sys.exit()
            if not PRIOR_MODIF:
                PRIOR_F = np.ones_like(CLASSES, dtype=np.float32)
        except IOError:
            tmp = np.zeros([NB_CL], dtype=np.float32)
            for k in range(NB_CL):
                tmp[k] = THRESH[CLASSES[k]]
            THRESH = tmp
            if VERB:
                print("No classes.thresh file found whereas probability thresholding is requested: using default thresholds")
    
    global POWERS
    if SINGLE_MASK:
        tmp = np.zeros([NB_CL], dtype=np.uint16)
        for k in range(NB_CL):
            tmp[k] = POWERS[CLASSES[k]]
        POWERS = tmp
        
    global CLASS_ABBRV
    tmp = []
    for k in range(NB_CL):
        tmp.append(CLASS_ABBRV[CLASSES[k]])
    CLASS_ABBRV = tmp

    global CLASS_NAMES
    tmp = []
    for k in range(NB_CL):
        tmp.append(CLASS_NAMES[CLASSES[k]])
    CLASS_NAMES = tmp
                
 
def main():
    """ Main function
    """

    # setup all parameters
    setup_params()

    config = tf.ConfigProto()
    global H_BACK
    if tf.test.is_gpu_available():
        H_BACK = "gpu"
        config.gpu_options.allow_growth = True
        print("MaxiMask is using GPU")
    else:
        H_BACK = "cpu"
        print("MaxiMask is using CPU")
        
    # open tf session first so all is done in one single session
    with tf.Session(config=config) as sess:
        nsaver = tf.train.import_meta_graph(NET_PATH + "/" + H_BACK + "_model.meta")
        nsaver.restore(sess, NET_PATH + "/model-150000")

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
    TNB_CL = 14
    TR_PRIORS = np.array([0.00212, 0.00900, 0.03678, 0.00033, 0.00033, 0.00860, 0.09170, 0.11298, 0.15486, 0.00149, 0.01316, 0.05940, 0.07312, 0.52271], dtype=np.float32)

    # positional command line argument
    IM_PATH = None

    # optional command line arguments
    NET_PATH = None
    PRIOR_MODIF = None
    PROBA_THRESH = None
    SINGLE_MASK = None
    BATCH_S = None
    VERB = None

    # default parameter values that can change depending on user specification (parameter files)
    CLASSES = np.arange(0, TNB_CL, dtype=np.int32)
    NB_CL = TNB_CL
    PRIORS = np.array([0.0002, 0.0008, 0.0080, 0.000035, 0.000035, 0.00001, 0.001, 0.0001, 0.0001, 0.0016, 0.013, 0.005, 0.07, 0.90], dtype=np.float32)
    PRIOR_F = (TR_PRIORS/(1-TR_PRIORS))*((1-PRIORS)/PRIORS)
    THRESH = np.array([0.52, 0.80, 0.82, 0.73, 0.98, 0.75, 0.70, 0.70, 0.59, 0.92, 0.57, 0.33, 0.70, 0.50], dtype=np.float32)

    tmp = np.arange(0, TNB_CL, dtype=np.uint16)
    tmp2 = 2*np.ones_like(tmp, dtype=np.uint16)
    tmp2[-1] = 0
    POWERS = np.power(tmp2, tmp)

    # other
    CLASS_ABBRV = ["CR", "HCL", "BCL", "HP", "BP", "P", "STL", "FR", "NEB", "SAT", "SP", "OV", "BBG", "BG"]
    CLASS_NAMES = ["CR: Cosmic Rays", "HCL: Hot Columns/Lines", "BCL: Bad Columns/Lines/Clusters", "HP: Hot Pixels", "BP: Bad Pixels", "P: Persistence", "STL: SaTeLlite trails", "FR: residual FRinging", "NEB: NEBulosities", "SAT: SATurated pixels", "SP: diffraction SPikes", "OV: Overscan", "BBG: Bright BackGround", "BG: BackGround"]

    H_BACK = None

    main()
