# MaxiMask
MaxiMask is a convolutional neural network that detects contaminants in astronomical images.

For now it is usable only with a tensorflow compatible GPU.

# Dependencies
* Python 2.7
* Scipy 1.0.1 or higher
* Astropy 2.0.7 or higher
* tensorflow-gpu 1.9 or higher

# Usage
The current version has 3 "modes" and should be used as following:
```
python MaxiMask.py <nn_path> <src_im_path> <optional batch_size>
```
Where:
* _nn_path_ is the path to the neural network save directory: by default it should be /path_to_repository/MaxiMask/model
* _src_im_path_ is the path to the image(s) to be processed
* _batch_size_ is an optional parameter to modify the batch size. Default is 16 but you might use a lower value like 8 or 4 if you don't have enough GPU RAM available.

The 3 modes depend on the _src_im_path_ parameter:
* if it is a file precising a specific HDU (CFITSIO notation) like <file.fits[nb_hdu]>, it will process only the hdu <nb_hdu> of <file.fits>. 
This should return a file <file.masks<nb_hdu>.fits> with the masks in the Primary HDU.
* if it is a file like <file.fits>, it will process all the HDUs that contain 2D data and copy the source image HDU otherwise.
This should return a file <file.masks.fits> that has the same HDU structure than <file.fits>.
* if it is a directory, it will process all the fits images of this directory as in the previous case.
This should return all the mask files in the same directory.

Processing an HDU consists in:
* estimating the sky background
* applying the dynamic compression
* slicing it into 400x400 sub images
* infering the 400x400 sub probability maps
* reconstructing the probability maps

# Class order:
* 1 CR: cosmic rays 
* 2 HC: hot columns
* 3 BC: bad columns
* 4 BL: bad lines
* 5 HP: hot pixels
* 6 BP: bad pixels
* 7 P: persistence
* 8 STL: satellite trails
* 9 FR: fringe patterns
* 10 NEB: nebulosities
* 11 SAT: saturated pixels
* 12 SP: diffraction spikes
* 13 BBG: bright background
* 14 BG: background

# LICENCE
Copyright (c) 2018 Maxime Paillassa. Both code and model weights are released under MIT licence. See LICENSE for details.
