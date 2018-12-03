# Copyright (c) 2018 Maxime Paillassa. Released under MIT.

import os
import sys
import math
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
    z = []
    for k in range(nb_blocks):
        x_b, x_e = x_l[k], min(x_l[k]+mesh_size, w)
        y_b, y_e = y_l[k], min(y_l[k]+mesh_size, h)

        mesh = im[y_b:y_e, x_b:x_e]
            
        tmp_mask = np.ones_like(mesh)
        idx = np.where(tmp_mask)
        init_std = np.std(mesh[idx[0], idx[1]])
        tmp_std = init_std
        tmp_med = np.median(mesh[idx[0], idx[1]])
        k = 3.0
        while True:
            to_keep = np.logical_and(tmp_mask*(mesh >= tmp_med-k*tmp_std), tmp_mask*(mesh <= tmp_med+k*tmp_std))
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
        z.append(b_v)

    # build the little mesh to median filter and to interpolate
    to_interp = np.zeros([h_blocks, w_blocks])
    for k in range(nb_blocks):
        to_interp[y_l[k]/mesh_size, x_l[k]/mesh_size] = z[k]
        
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


def process_batch(src_im, masks, batch_s, tot_l, first_p, last_p, sess, IM_SIZE, NB_CL):
    """ Process one batch
    """
    
    IM4 = IM_SIZE/4
    h,w = src_im.shape

    tmp_masks = np.zeros([batch_s, IM_SIZE, IM_SIZE, NB_CL])
    inp = np.zeros([batch_s, IM_SIZE, IM_SIZE])

    # prepare inputs and make inference
    k = 0
    for coord in tot_l[first_p:last_p]:
        inp[k] = src_im[coord[1]:coord[1]+IM_SIZE, coord[0]:coord[0]+IM_SIZE]
        k += 1
    tmp_masks = sess.run("predictions:0", {"rinputs:0": np.reshape(inp, [batch_s, IM_SIZE, IM_SIZE, 1])})

    # copy in final mask
    k = 0
    for x,y in tot_l[first_p:last_p]:
        masks[IM4+y:y+IM_SIZE-IM4, IM4+x:x+IM_SIZE-IM4, :] = tmp_masks[k][IM4:IM_SIZE-IM4, IM4:IM_SIZE-IM4]
        if x==0:
            masks[IM4+y:y+IM_SIZE-IM4, x:x+IM4, :] = tmp_masks[k][IM4:IM_SIZE-IM4, :IM4]
        if y==0:
            masks[y:y+IM4, IM4+x:x+IM_SIZE-IM4, :] = tmp_masks[k][:IM4, IM4:IM_SIZE-IM4]
        if x==w-IM_SIZE:
            masks[IM4+y:y+IM_SIZE-IM4, x+IM_SIZE-IM4:x+IM_SIZE, :] = tmp_masks[k][IM4:IM_SIZE-IM4, IM_SIZE-IM4:IM_SIZE]
        if y==h-IM_SIZE:
            masks[y+IM_SIZE-IM4:y+IM_SIZE, IM4+x:x+IM_SIZE-IM4, :] = tmp_masks[k][IM_SIZE-IM4:IM_SIZE, IM4:IM_SIZE-IM4]
        if x==0 and y==0:
            masks[y:y+IM4, x:x+IM4, :] = tmp_masks[k][:IM4, :IM4]
        if x==0 and y==h-IM_SIZE:
            masks[y+IM_SIZE-IM4:y+IM_SIZE, x:x+IM4, :] = tmp_masks[k][IM_SIZE-IM4:IM_SIZE, :IM4]
        if x==w-IM_SIZE and y==0:
            masks[y:y+IM4, x+IM_SIZE-IM4:x+IM_SIZE, :] = tmp_masks[k][:IM4, IM_SIZE-IM4:IM_SIZE]
        if x==w-IM_SIZE and y==h-IM_SIZE:
            masks[y+IM_SIZE-IM4:y+IM_SIZE, x+IM_SIZE-IM4:x+IM_SIZE, :] = tmp_masks[k][IM_SIZE-IM4:IM_SIZE, IM_SIZE-IM4:IM_SIZE]
        k += 1


def process_hdu(src_im, masks, sess, max_b, IM_SIZE, NB_CL):
    """ Process one hdu 
    """

    IM2 = IM_SIZE/2
    IM4 = IM_SIZE/4

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
    if len(tot_l)<=max_b:
        process_batch(src_im, masks, len(tot_l), tot_l, 0, len(tot_l), sess, IM_SIZE, NB_CL)
    # otherwise iterate over all batches to do
    else:
        nb_step = len(tot_l)/max_b+1
        for st in range(nb_step-1):
            if st<nb_step-1:
                process_batch(src_im, masks, max_b, tot_l, st*max_b, (st+1)*max_b, sess, IM_SIZE, NB_CL)
        # manage the last (incomplete) batch
        re = len(tot_l)-(nb_step-1)*max_b
        if re:
            process_batch(src_im, masks, re, tot_l, (nb_step-1)*max_b, len(tot_l), sess, IM_SIZE, NB_CL)


def process_file(src_im_path, sess, max_b, IM_SIZE, NB_CL):
    """ Process one fits file
    If the hdu is specified in the name of the file by <[nb]> it processes the hdu <nb>
    Otherwise it processes all the hdus containing data
    """

    if src_im_path[-1]=="]":
        # process the specified hdu
        spec_hdu = src_im_path.split("[")[1].split("]")[0]
        n = int(len(spec_hdu)+2)
        with fits.open(src_im_path[:-n]) as src_im_hdu:
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

        masks = np.zeros([h,w,NB_CL], dtype=np.float32)
        process_hdu(src_im, masks, sess, max_b, IM_SIZE, NB_CL)
        hdu = fits.PrimaryHDU(np.transpose(masks, (2,0,1)))
        hdu.writeto(src_im_path[:-n].replace(".fits", ".masks" + spec_hdu + ".fits"), overwrite=True)
    else:
        # process all hdus containing data
        with fits.open(src_im_path) as src_im_hdu:
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

                    masks = np.zeros([h,w,NB_CL], dtype=np.float32)
                    process_hdu(src_im, masks, sess, max_b, IM_SIZE, NB_CL)
                    if k==0:
                        m_hdu = fits.PrimaryHDU(np.transpose(masks, (2,0,1)))
                        hdu.append(m_hdu)
                    else:
                        sub_hdu = fits.ImageHDU(np.transpose(masks, (2,0,1)))
                        hdu.append(sub_hdu)
                else:
                    # if this seems not to be data then copy the hdu
                    tmp_hdu = src_im_hdu[k]
                    hdu.append(tmp_hdu)

            hdu.writeto(src_im_path.replace(".fits", ".masks.fits"), overwrite=True)


def main():
    IM_SIZE = 400
    NB_CL = 14

    if len(sys.argv)!=4 and len(sys.argv)!=5:
        print "Usage: python " + sys.argv[0] + " <cpu|gpu> <nn_path> <src_im_path> <batch_s>"
        print "Where: "
        print "    cpu|gpu is a string speficifying if you are using CPU or GPU"
        print "    nn_path is the path to the neural network save directory"
        print "    src_im_path is the path to the image(s) to be processed"
        print "    batch_s is the batch size for inference (optional, default is 8)"
        sys.exit()

    hard_backend = sys.argv[1]
    net_path = sys.argv[2]
    src_im_path = sys.argv[3]
    if len(sys.argv)==5:
        max_b = int(sys.argv[4])
    else:
        max_b = 8

    # gpu options
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    
    # open tf session first so all is done in one single session
    with tf.Session(config=config) as sess:
        nsaver = tf.train.import_meta_graph(net_path + "/" + hard_backend + "_model.meta")
        nsaver.restore(sess, net_path + "/model-150000")
        
        if os.path.isfile(src_im_path) or src_im_path[-1]=="]":
            # process the image
            process_file(src_im_path, sess, max_b, IM_SIZE, NB_CL)
        else:
            # process all the images of the directory
            for src_im_s in os.listdir(src_im_path):
                if "fits" in src_im_s:
                    process_file(src_im_path + "/" + src_im_s, sess, max_b, IM_SIZE, NB_CL)


if __name__=="__main__":
    main()
