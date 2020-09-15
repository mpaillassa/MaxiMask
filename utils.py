import math
import time
import numpy as np
import scipy.interpolate as interp
import scipy.signal as sign
from astropy.io import fits


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
    zs = [None]*nb_blocks
    for b in range(nb_blocks):
        x_b, x_e = x_l[b], min(x_l[b]+mesh_size, w)
        y_b, y_e = y_l[b], min(y_l[b]+mesh_size, h)

        mesh = im[y_b:y_e, x_b:x_e]
        
        if np.all(mesh==0):
            z[b] = 0
        else:
            tmp_mask = np.ones_like(mesh)
            idx = np.where(tmp_mask)
            init_std = np.std(mesh[idx[0], idx[1]])
            tmp_std = init_std
            tmp_med = np.median(mesh[idx[0], idx[1]])
            k = 3.0
            while True:
                to_keep = np.logical_and(tmp_mask*(mesh >= tmp_med-k*tmp_std), tmp_mask*(mesh <= tmp_med+k*tmp_std))
                to_keep = np.logical_and(to_keep, np.logical_not(mesh<=0))
                if np.all(to_keep==tmp_mask):
                    break
                else:
                    tmp_mask *= to_keep
                    idx = np.where(tmp_mask)
                    tmp_std = np.std(mesh[idx[0], idx[1]])
                    tmp_med = np.median(mesh[idx[0], idx[1]])
            idx = np.where(tmp_mask)
            b_v = np.mean(mesh[idx[0], idx[1]])
            z[b] = b_v
            zs[b] = np.std(mesh[idx[0], idx[1]])

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

    ## for sigma

    # build the little mesh to median filter and to interpolate
    to_interp = np.zeros([h_blocks, w_blocks])
    for b in range(nb_blocks):
        to_interp[int(y_l[b]/mesh_size), int(x_l[b]/mesh_size)] = zs[b]
        
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
    sig_map = f(np.arange(h), np.arange(w)).astype(np.float32)

    return back_val, sig_map


@timeit
def dynamic_compression(im):
    """ Dynamical compressing: sky background subtraction and sigma normalizing
    """

    # just a check to remove eventual nan values
    np.place(im, np.isnan(im), 0)
    np.place(im, im>80000, 80000)

    # dynamic compression
    bg_map, si_map = background_est(im)
    np.place(bg_map, np.isnan(bg_map), 0)

    im -= bg_map
    im /= si_map
