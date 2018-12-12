# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import os
import sys
import math
import argparse
import numpy as np
import scipy.interpolate as interp
import scipy.signal as sign
from astropy.io import fits

import tensorflow as tf


def background_est(im):
    """Process a Sextractor like background estimation
    """
    
    h,w = im.shape
    mesh_size = 200

    # get the number blocks to process
    h_blocks = (h-h%mesh_size)/mesh_size
    w_blocks = (w-w%mesh_size)/mesh_size
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
        to_interp[y_l[b]/mesh_size, x_l[b]/mesh_size] = z[b]
        
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
    back_val = f(np.arange(h), np.arange(w))

    return back_val 


def process_batch(src_im, results, sess, batch_s, tot_l, first_p, last_p):
    """ Process one batch
    """
    
    h,w = src_im.shape

    inp = np.zeros([batch_s, IM_SIZE, IM_SIZE], dtype=np.float32)
    tmp_results = np.zeros([batch_s, IM_SIZE, IM_SIZE, NB_CL], dtype=np.float32)

    # prepare inputs and make inference
    k = 0
    for coord in tot_l[first_p:last_p]:
        inp[k] = src_im[coord[1]:coord[1]+IM_SIZE, coord[0]:coord[0]+IM_SIZE]
        k += 1
    tmp_results = sess.run("predictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1])})[:,:,:,CLASSES]

    # copy in final mask
    k = 0
    for x,y in tot_l[first_p:last_p]:
        results[IM4+y:y+IM_SIZE-IM4, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][IM4:IM_SIZE-IM4, IM4:IM_SIZE-IM4]
        if x==0:
            results[IM4+y:y+IM_SIZE-IM4, x:x+IM4] = tmp_results[k][IM4:IM_SIZE-IM4, :IM4]
        if y==0:
            results[y:y+IM4, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][:IM4, IM4:IM_SIZE-IM4]
        if x==w-IM_SIZE:
            results[IM4+y:y+IM_SIZE-IM4, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][IM4:IM_SIZE-IM4, IM_SIZE-IM4:IM_SIZE]
        if y==h-IM_SIZE:
            results[y+IM_SIZE-IM4:y+IM_SIZE, IM4+x:x+IM_SIZE-IM4] = tmp_results[k][IM_SIZE-IM4:IM_SIZE, IM4:IM_SIZE-IM4]
        if x==0 and y==0:
            results[y:y+IM4, x:x+IM4] = tmp_results[k][:IM4, :IM4]
        if x==0 and y==h-IM_SIZE:
            results[y+IM_SIZE-IM4:y+IM_SIZE, x:x+IM4] = tmp_results[k][IM_SIZE-IM4:IM_SIZE, :IM4]
        if x==w-IM_SIZE and y==0:
            results[y:y+IM4, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][:IM4, IM_SIZE-IM4:IM_SIZE]
        if x==w-IM_SIZE and y==h-IM_SIZE:
            results[y+IM_SIZE-IM4:y+IM_SIZE, x+IM_SIZE-IM4:x+IM_SIZE] = tmp_results[k][IM_SIZE-IM4:IM_SIZE, IM_SIZE-IM4:IM_SIZE]
        k += 1


def process_hdu(src_im, results, sess):
    """ Process one hdu 
    """

    h,w = src_im.shape
    if h<IM_SIZE or w<IM_SIZE:
        print "One of the two image dimension is less than " + str(IM_SIZE) + " : not supported yet"
        print "Exiting..."
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
        nb_step = len(tot_l)/BATCH_S+1
        for st in range(nb_step-1):
            if st<nb_step-1:
                process_batch(src_im, results, sess, BATCH_S, tot_l, st*BATCH_S, (st+1)*BATCH_S)
        # manage the last (incomplete) batch
        re = len(tot_l)-(nb_step-1)*BATCH_S
        if re:
            process_batch(src_im, results, sess, re, tot_l, (nb_step-1)*BATCH_S, len(tot_l))

    if PRIOR_MODIF:
        cl = 0
        for tcl in range(TNB_CL):
            if CLASSES[tcl]:
                prior_factor = (TR_PRIORS[tcl]*(1-PRIORS[tcl]))/((1-TR_PRIORS[tcl])*PRIORS[tcl])
                results[:,:,cl] = results[:,:,cl]/(results[:,:,cl] + prior_factor*(1-results[:,:,cl]))
                cl += 1
                
    if PROBA_THRESH:
        thresholds = THRESH[CLASSES]
        results[(results > thresholds)] = 1
        results[(results <= thresholds)] = 0


def process_file(sess, src_im_s):
    """ Process one fits file
    If the hdu is specified in the name of the file by <[nb]> it processes the hdu <nb>
    Otherwise it processes all the hdus containing data
    """
    
    if src_im_s!="":
        im_path = IM_PATH + "/" + src_im_s
    else:
        im_path = IM_PATH

    if im_path[-1]=="]":
        # process the specified hdu
        spec_hdu = im_path.split("[")[1].split("]")[0]
        n = int(len(spec_hdu)+2)
        with fits.open(im_path[:-n]) as src_im_hdu:
            if int(spec_hdu)>len(src_im_hdu):
                print "Error: requesting hdu " + spec_hdu + " when image has only " + str(len(src_im_hdu)) + " hdu(s)"
                print "Exiting..."
                sys.exit()

            src_im = src_im_hdu[int(spec_hdu)].data.astype(np.float32)
            h,w = src_im.shape

        # dynamic compression
        bg_map = background_est(src_im)
        src_im -= bg_map
        sig = np.std(src_im)
        src_im /= sig
        if VERB: print IM_PATH + " dynamic compression done"

        results = np.zeros([h,w,NB_CL], dtype=np.float32)
        process_hdu(src_im, results, sess)
        if VERB: print IM_PATH + " inference done"
        if PROBA_THRESH:
            hdu = fits.PrimaryHDU(np.transpose(results, (2,0,1)).astype(np.uint8))
        else:
            hdu = fits.PrimaryHDU(np.transpose(results, (2,0,1)))
        hdu.writeto(im_path[:-n].split(".fits")[0] + ".masks" + spec_hdu + ".fits", overwrite=True)
    else:
        # process all hdus containing data
        with fits.open(im_path) as src_im_hdu:
            nb_hdu = len(src_im_hdu)
            hdu = fits.HDUList()
            for k in range(nb_hdu):
                if len(src_im_hdu[k].shape)==2:
                    src_im = src_im_hdu[k].data.astype(np.float32)
                    h,w = src_im.shape

                    # dynamic compression
                    bg_map = background_est(src_im)
                    src_im -= bg_map
                    sig = np.std(src_im)
                    src_im /= sig
                    if VERB: print "HDU " + str(k) + "/" + str(nb_hdu-1) + " dynamic compression done"

                    results = np.zeros([h,w,NB_CL], dtype=np.float32)
                    process_hdu(src_im, results, sess)
                    if VERB: print "HDU " + str(k) + "/" + str(nb_hdu-1) + " inference done"
                    if k==0:
                        if PROBA_THRESH:
                            m_hdu = fits.PrimaryHDU(np.transpose(results, (2,0,1)).astype(np.uint8))
                        else:
                            m_hdu = fits.PrimaryHDU(np.transpose(results, (2,0,1)))
                        hdu.append(m_hdu)
                    else:
                        if PROBA_THRESH:
                            sub_hdu = fits.ImageHDU(np.transpose(results, (2,0,1)).astype(np.uint8))
                        else:
                            sub_hdu = fits.ImageHDU(np.transpose(results, (2,0,1)))
                        hdu.append(sub_hdu)
                else:
                    # if this seems not to be data then copy the hdu
                    hdu.append(src_im_hdu[k])
                if VERB: print "HDU " + str(k) + "/" + str(nb_hdu-1) + " done"
           
            if VERB: "Writing " + im_path.split(".fits")[0] + ".masks.fits to disk"
            hdu.writeto(im_path.split(".fits")[0] + ".masks.fits", overwrite=True)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_params():
    """
    """

    # parameter parser
    parser = argparse.ArgumentParser(description='MaxiMask command line parameters:')

    # necessary parameters
    parser.add_argument("hard_back", type=str, help='<cpu> or <gpu> depending on your tensorflow installation hardware backend', choices=('cpu', 'gpu'))
    parser.add_argument("im_path", type=str, help='path the image(s) to be processed')

    # optional parameters
    parser.add_argument("--net_path", type=str, help='path to the neural network graphs and weights directory. Default is </abs_path_to_rep/model>', default=os.path.dirname(os.path.abspath(__file__)) + "/model")
    parser.add_argument("--prior_modif", type=str2bool, help='bool indicating if probability maps should be prior modified. Default is True', default=True)
    parser.add_argument("--proba_thresh", type=str2bool, help='bool indicating if probability maps should be thresholded. Default is True', default=True)
    parser.add_argument("--batch_size", type=int, help='neural network batch size. Default is 8. You might want to use a lower value if you have RAM issues', default=8)
    parser.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")

    # read arguments
    args = parser.parse_args()

    global HARD_BACK
    global IM_PATH

    global NET_PATH
    global PRIOR_MODIF
    global PROBA_THRESH
    global BATCH_S
    global VERB

    HARD_BACK = args.hard_back
    IM_PATH = args.im_path

    NET_PATH = args.net_path
    PRIOR_MODIF = args.prior_modif
    PROBA_THRESH = args.proba_thresh
    BATCH_S = args.batch_size
    VERB = args.verbose

    # read classes.flags
    global CLASSES
    try:
        with open("classes.flags") as fd:
            lines = fd.readlines()
            if len(lines)!=TNB_CL:
                print "Error: classes.flags must contain " + str(TNB_CL) + " lines, one for each class"
                print "Exiting..."
                sys.exit()
            for k in range(TNB_CL):
                try:
                    if int(lines[k].split()[1]):
                        CLASSES[k] = True
                    else:
                        CLASSES[k] = False
                except (ValueError, IndexError) as e:
                    print "Error: classes.flags file exists but may not be well formatted (at least one class does not have a proper boolean flag)"
                    print "Exiting..."
                    sys.exit()
            global NB_CL
            NB_CL = np.sum(CLASSES)
    except IOError:
        if VERB:
            print "No classes.flags found: MaxiMask will output probabilities/masks for all classes"
    
    # if prior modif is requested read classes.priors
    if PRIOR_MODIF:
        global PRIORS
        try:
            with open("classes.priors") as fd:
                lines = fd.readlines()
                if len(lines)!=TNB_CL:
                    print "Error: classes.priors must contain " + str(TNB_CL) + " lines, one for each class"
                    print "Exiting..."
                    sys.exit()
                for k in range(TNB_CL):
                    if CLASSES[k]:
                        try:
                            PRIORS[k] = float(lines[k].split()[1])
                        except (ValueError, IndexError) as e:
                            print "Error: classes.priors file exists but may not be well formatted (at least one requested class does not have a proper prior)"
                            print "Exiting..."
                            sys.exit()
        except IOError:
            if VERB:
                print "No classes.priors file found whereas prior modification is requested: using default priors"
    
    # if probability thresholding is requested read classes.thresh
    if PROBA_THRESH:
        global THRESH
        try:
            with open("classes.thresh") as fd:
                lines = fd.readlines()
                if len(lines)!=TNB_CL:
                    print "Error: classes.thresh must contain " + str(TNB_CL) + " lines, one for each class"
                    print "Exiting..."
                    sys.exit()
                for k in range(TNB_CL):
                    if CLASSES[k]:
                        try:
                            THRESH[k] = float(lines[k].split()[1])
                        except (ValueError, IndexError) as e:
                            print "Error: classes.thresh file exists but may not be well formatted (at least one requested class does not have a proper threshold)"
                            print "Exiting..."
                            sys.exit()
        except IOError:
            if VERB:
                print "No classes.thresh file found whereas probability thresholding is requested: using default thresholds"
    
    
def main():
    
    # setup all parameters
    setup_params()

    # gpu options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    # open tf session first so all is done in one single session
    with tf.Session(config=config) as sess:
        nsaver = tf.train.import_meta_graph(NET_PATH + "/" + HARD_BACK + "_model.meta")
        nsaver.restore(sess, NET_PATH + "/model")
        
        if os.path.isfile(IM_PATH) or IM_PATH[-1]=="]":
            # process the image
            if VERB: print "Processing " + IM_PATH
            process_file(sess, "")
            if VERB: print IM_PATH + " done"
        else:
            # process all the images of the directory
            if VERB: print "Processing " + IM_PATH
            for src_im_s in os.listdir(IM_PATH):
                if "fits" in src_im_s:
                    if VERB: print "Processing " + IM_PATH + "/" + src_im_s
                    process_file(sess, src_im_s)
                    if VERB: print IM_PATH + "/" + src_im_s + " done"


if __name__=="__main__":
    # parameter values that should never change and should not be changed by user
    IM_SIZE = 400
    IM2 = IM_SIZE/2
    IM4 = IM_SIZE/4
    TNB_CL = 14
    TR_PRIORS = np.array([0.00224, 0.00803, 0.00796, 0.00800, 0.00035, 0.00035, 0.00911, 0.07027, 0.12188, 0.16426, 0.00161, 0.01409, 0.07717, 0.58629])

    # parameter which values are read from command line
    HARD_BACK = None
    IM_PATH = None

    NET_PATH = None
    PRIOR_MODIF = None
    PROBA_THRESH = None
    BATCH_S = None
    VERB = None

    # default parameter values that can change depending on user specification 
    CLASSES = np.ones([TNB_CL], dtype=np.bool)
    NB_CL = np.sum(CLASSES)
    PRIORS = np.array([0.0002, 0.0008, 0.0008, 0.0008, 0.000035, 0.000035, 0.00001, 0.001, 0.00001, 0.00001, 0.0016, 0.014, 0.07, 0.90])
    THRESH = np.array([0.61, 0.61, 0.61, 0.56, 0.59, 0.96, 0.91, 0.45, 0.83, 0.46, 0.92, 0.76, 0.53, 0.52])

    main()
