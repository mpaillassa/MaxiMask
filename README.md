# MaxiMask
MaxiMask is a convolutional neural network (CNN) that detects contaminants in astronomical images.

<p align="center">
  <img src="logo.png" width="300">
</p>

# Dependencies
* Python 2 or 3
* Scipy >=1.0.1
* Astropy >=2.0.7
* tensorflow or tensorflow-gpu >=1.9 (CPU is expected to be much slower than GPU)

(Older versions may work but it has not been tested)
# Usage

## Minimal use
The minimal way to run MaxiMask is:
```
./maximask.py <im_path>
```
Where <im_path> indicates the images you want to process. It can specify:
  - A specific image HDU (CFITSIO notation) like <file.fits[nb_hdu]>: MaxiMask will process only the hdu <nb_hdu> of <file.fits>. 
This should return a file <file.masks<nb_hdu>.fits> with the masks in the Primary HDU.
  - A fits file like <file.fits>: MaxiMask will process all the image HDUs that contain 2D data and copy the source HDU otherwise.
This should return a file <file.masks.fits> that has the same HDU structure than <file.fits>.
  - A directory: MaxiMask will process all the fits images of this directory as in the previous case.
This should return all the mask files in the same directory. 
  - A list file: this must be a file with <.list> extension containing one fits file path </path/to/file.fits> per line. MaxiMask will process each file as in the second case. 

You can add the repository path to your PATH variable to use it anywhere in your machine (add the following line to your .bashrc to make it permanent):
```
export PATH=$PATH:/path/to/MaxiMask/repository
```
You can also create a symbolic link using the following command in the MaxiMask repository directory:
```
ln -sf maximask.py maximask
```
So that you can just run ```maximask <im_path>``` from anywhere in your machine.

## Minimal example
If you run:
```
maximask test_im.fits.fz
```
You should obtain a file named <test_im.masks.fits> that has the same content as <test_out.fits.fz>.

## General use
Here is full description of MaxiMask. It can be obtained by running ```maximask -h```
```
usage: maximask [-h] [--net_path NET_PATH] [--prior_modif PRIOR_MODIF]
                [--proba_thresh PROBA_THRESH] [--single_mask SINGLE_MASK]
                [--batch_size BATCH_SIZE] [-v]
                im_path

MaxiMask command line parameters:

positional arguments:
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
  --single_mask SINGLE_MASK
                        bool indicating if resulting masks are joined in a
                        single mask using powers of two
  --batch_size BATCH_SIZE
                        neural network batch size. Default is 8. You might
                        want to use a lower value if you have RAM issues
  -v, --verbose         activate output verbosity
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

### Single mask
If this option is required, MaxiMask will return only one mask by compiling each requested class using power of 2. Each class can be identified with its power of two. 

### Class selection
Selecting some specific classes can be done using a file named _classes.flags_ where one can indicate which classes are requested with 0 and 1. Example of the required syntax is given is _classes.flags_.  
MaxiMask will automatically look for _classes.flags_. If it does not exist, MaxiMask will output probability maps/binary maps/single mask for all classes.  
Depending on what is returned, the output fits header will be filled with corresponding informations.

### File syntax and class names 
For more convenience when modifying _classes.flags_, _classes.priors_ or _classes.thresh_, the syntax choice has been to use two space separated columns:
1. the abbreviated class names.
2. the values of interest.

That is something like this:
```
CR  <flag|prior|threshold>
HCL <flag|prior|threshold>
DCL <flag|prior|threshold>
HP  <flag|prior|threshold>
DP  <flag|prior|threshold>
P   <flag|prior|threshold>
STL <flag|prior|threshold>
FR  <flag|prior|threshold>
NEB <flag|prior|threshold>
SAT <flag|prior|threshold>
SP  <flag|prior|threshold>
OV  <flag|prior|threshold>
BBG <flag|prior|threshold>
BG  <flag|prior|threshold>
```

This is the required syntax. If not respected while reading such a file, MaxiMask will exit with an appropriate error message.  
(Note that _classes.priors_ and _classes.thresh_ should contain one line per class even when not all classes are requested; lines of non requested classes will just be ignored).

Abbreviated names stand for:

| Abbreviated name | Full name | Binary Code |
| --- | --- | --- |
| CR | Cosmic Rays | 1 |
| HCL | Hot Columns/Lines | 2 |
| DCL | Dead Columns/Lines/Clusters | 4 |
| HP | Hot Pixels | 8 |
| DP | Dead Pixels | 16 |
| P | Persistence | 32 |
| STL | SaTeLlite trails | 64 |
| FR | FRinge patterns | 128 |
| NEB | NEBulosities | 256 |
| SAT | SATurated pixels | 512 |
| SP | diffraction SPikes | 1024 |
| OV | OVerscanned pixels | 2048 |
| BBG | Bright BackGround pixel | 4096 |
| BG | Background | 0 |

Each power of two is the corresponding single mask code of the class.

# MaxiTrack

MaxiTrack behaves just like MaxiMask: it can process images using the same formats (specific HDU, specific image, directory or list file):
```
./maxitrack.py <im_path>
```
You may use the same procedure than MaxiMask to use it from anywhere in your machine.

## Minimal example
If you run:
```
maxitrack test_im.fits.fz
```
You should obtain a file named <maxitrack.out> containing the line:
```
test_im.fits.fz 0.019503068732568823
```
The number corresponding to the image name is the probability that this image is affected by tracking error.
When running again, MaxiTrack will always append the new results to this file <maxitrack.out>. 

## General use
Here is full description of MaxiMask. It can be obtained by running ```maxitrack -h```
```
usage: maxitrack.py [-h] [--net_path NET_PATH] [--prior_value PRIOR_VALUE]
                    [--frac FRAC] [--batch_size BATCH_SIZE] [-v]
                    im_path

MaxiTrack command line parameters:

positional arguments:
  im_path               path the image(s) to be processed

optional arguments:
  -h, --help            show this help message and exit
  --net_path NET_PATH   path to the neural network graphs and weights
                        directory. Default is </abs_path_to_rep/model>
  --prior_value PRIOR_VALUE
                        float defining the expected prior in data. Default is
                        0.05
  --frac FRAC           int defining the number of HDU to use. Default is -1,
                        meaning that MaxiTrack will use all HDU
  --batch_size BATCH_SIZE
                        neural network batch size. Default is 8. You might
                        want to use a lower value if you have RAM issues
  -v, --verbose         activate output verbosity
```

### Probability prior modification
As in MaxiMask, priors can be specified to adjust the output probabilities to new expected class proportions. As there are only two classes in MaxiTrack (tracking or not tracking), only one prior corresponding to the expected proportion of images affected by tracking errors has to be speficied. Default is 0.05, i.e 5% of images affected by tracking errors.

### Fraction option
When giving a FITS file containing N HDUs, MaxiTrack will by default use the N HDUs to compute the output probability for the whole field. In order to run MaxiTrack faster, you can specify a number FRAC<N of HDUs to use to compute the output probability.


# LICENSE
Copyright (c) 2018 Maxime Paillassa. 

Both code and model weights are released under MIT license. 

See LICENSE for details.
