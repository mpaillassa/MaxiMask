# MaxiMask
MaxiMask is a convolutional neural network (CNN) that detects contaminants in astronomical images.

# Dependencies
* Python 2.7
* Scipy >=1.0.1
* Astropy >=2.0.7
* tensorflow or tensorflow-gpu >=1.9

(Older versions may work but it has not been tested)
# Usage

## Minimal use
The minimal way to run MaxiMask is:
```
python MaxiMask.py <cpu|gpu> <im_path>
```
Where:
* <cpu|gpu> indicates your tensorflow installation hardware backend. It should always be "cpu" or "gpu". CPU is expected to be much slower than GPU.
* <im_path> indicates the images you want to process. It can specify:
  - A specific image HDU (CFITSIO notation) like <file.fits[nb_hdu]>: MaxiMask will process only the hdu <nb_hdu> of <file.fits>. 
This should return a file <file.masks<nb_hdu>.fits> with the masks in the Primary HDU.
  - A fits file like <file.fits>: MaxiMask will process all the image HDUs that contain 2D data and copy the source HDU otherwise.
This should return a file <file.masks.fits> that has the same HDU structure than <file.fits>.
  - A directory: MaxiMask will process all the fits images of this directory as in the previous case.
This should return all the mask files in the same directory. 

## Minimal example
If you run:
```
python MaxiMask.py <cpu|gpu> test_im.fits.fz
```
You should obtain a file named <test_im.masks.fits> that has the content as <test_out.fits.fz>.

## General use
Here is full description of MaxiMask. You can obtain it by running ``` python MaxiMask.py -h```
```
usage: MaxiMask.py [-h] [--net_path NET_PATH] [--prior_modif PRIOR_MODIF]  
                   [--proba_thresh PROBA_THRESH] [--batch_size BATCH_SIZE]  
                   [-v]  
                   {cpu,gpu} im_path  

MaxiMask command line parameters:

positional arguments:
  {cpu,gpu}             <cpu> or <gpu> depending on your tensorflow
                        installation hardware backend
  im_path               path the image(s) to be processed

optional arguments:
  -h, --help            show this help message and exit
  --net_path NET_PATH   path to the neural network graphs and weights
                        directory. Default is </abs_path_to_rep/model>
  --prior_modif PRIOR_MODIF
                        bool indicating if probability maps should be prior
                        modified. Default is True
  --proba_thresh PROBA_THRESH
                        bool indicating if probability maps should be
                        thresholded. Default is True
  --batch_size BATCH_SIZE
                        neural network batch size. Default is 8. You might
                        want to use a lower value if you have RAM issues
  -v, --verbose         increase output verbosity
```

The CNN outputs are probability maps for each class.  
By default MaxiMask will prior adjust and threshold these probabilities with default parameters.

### Probability prior modification
The prior modification aims to modify the MaxiMask output probabilities to match new priors, i.e new class proportions.  
When it is requested (default behaviour), MaxiMask will look for a file named _classes.priors_ containing the new priors.  
If prior modification is requested and this file does not exist, it will use default priors indicated in the example file _classes.priors_, which also shows the required syntax.

### Probability thresholding
The probability thresholding aims to threshold the MaxiMask output probabilities to obtain uint8 maps instead of float32 maps. One can use various thresholds to trade off true positive rate vs false positive rate.   
When it is requested (default behaviour), MaxiMask will look for a file named _classes.thresh_ containing the thresholds.  
If probability thresholding is requested and this file does not exist, it will use default thresholds indicated in the example file _classes.thresh_, which also shows the required syntax.

### Class selection
Selecting some specific classes can be done using a file named _classes.flags_ where one can indicate which classes are requested with 0 and 1. Example of the required syntax is given is _classes.flags_.  
MaxiMask will automatically look for _classes.flags_. If it does not exist, MaxiMask will output probability/binary maps for all classes.

### File syntax and class names 
For more convenience when modifying _classes.flags_, _classes.priors_ or _classes.thresh_, the syntax choice has been to use two space separated columns:
1. the abbreviated class names.
2. the values of interest.

This is the required syntax. If not respected while reading such a file, MaxiMask will exit with an appropriate error message.  
(Note that _classes.priors_ and _classes.thresh_ should contain one line per class even when not all classes are requested; lines of non requested classes will just be ignored).

Abbreviated names stand for:
* CR: cosmic rays 
* HC: hot columns
* BC: bad columns
* BL: bad lines
* HP: hot pixels
* BP: bad pixels
* P: persistence
* STL: satellite trails
* FR: fringe patterns
* NEB: nebulosities
* SAT: saturated pixels
* SP: diffraction spikes
* BBG: bright background
* BG: background

# LICENSE
Copyright (c) 2018 Maxime Paillassa. 

Both code and model weights are released under MIT license. 

See LICENSE for details.
