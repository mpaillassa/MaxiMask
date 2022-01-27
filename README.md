[![DOI](https://zenodo.org/badge/156887999.svg)](https://zenodo.org/badge/latestdoi/156887999)

# MaxiMask and MaxiTrack
MaxiMask and MaxiTrack are convolutional neural networks (CNNs) that can detect contaminants in astronomical images. They relate to the following publication:
<img align="right" width="100" src="https://github.com/mpaillassa/MaxiMask/blob/master/imgs/logo.png">

A&A: https://doi.org/10.1051/0004-6361/201936345 

arXiv: https://arxiv.org/abs/1907.08298

MaxiMask is a semantic segmentation CNN (identifying contaminants at the pixel level) that can detect various contaminants, including trails, as shown in the image below where it detects the Starlink satellites in the famous DECam image.

<p align="center">
  <img src="imgs/https://github.com/mpaillassa/MaxiMask/blob/master/imgs/starlink1.gif" width="600">
</p>

MaxiTrack is an image classification CNN (identifying contaminants at the image level) that can detect tracking errors.

# Installation

You can install the latest version of MaxiMask and MaxiTrack at once via pip:
```
pip install MaxiMask -U
```

Be aware that additional librairies are needed to enable GPU support with tensorflow (CUDA, CuDNN). You can check [here](https://www.tensorflow.org/install/gpu) for more information. 

# Usage

## Minimal use
The minimal way to run MaxiMask is:
```
maximask <im_path>
```
Where `im_path` indicates the images you want to process. It can specify:
  - A specific image HDU (CFITSIO notation) like `file.fits[nb_hdu]`: MaxiMask will process only the hdu `nb_hdu` of `file.fits`. 
This should return a file `file.masks<nb_hdu>.fits` with the masks in the Primary HDU.
  - A fits file like `file.fits`: MaxiMask will process all the image HDUs that contain 2D data and copy the source HDU otherwise.
This should return a file `file.masks.fits` that has the same HDU structure than <file.fits>.
  - A directory: MaxiMask will process all the fits images of this directory as in the previous case.
  - A list file: this must be a file with <.list> extension containing one fits file path </path/to/file.fits> per line. MaxiMask will process each file as in the second case.
 
Note that you can also provide `.fits.fz` or `.fits.gz` extensions to MaxiMask and that the resulting masks are written in the same directory than the input image(s) designed by `im_path`.

## Minimal example
If you run:
```
maximask test/test_im.fits.fz -v
```
You should obtain a file named <test_im.masks.fits> in the <test> directory that is the same as <test_out.fits.fz>. It consists of the binary masks of each contaminant class. You can find the list of contaminant classes below in this README.

Note that the first run is always slower because tensorflow proceeds to some optimizations during the first pass. You can experiment this by running:
```
maximask test/test.list -v
```
Which will process the test image two times in a row.

## General use
Here is a more comprehensive description of MaxiMask. It can be obtained by running `maximask -h`:
```
usage: maximask [-h] [--net_dir NET_DIR] [--config_dir CONFIG_DIR]
                [--prior_modif PRIOR_MODIF] [--proba_thresh PROBA_THRESH]
                [--single_mask SINGLE_MASK] [--batch_size BATCH_SIZE] [-v]
                im_path

MaxiMask command line parameters:

positional arguments:
  im_path               path to the image(s) to be processed

optional arguments:
  -h, --help            show this help message and exit
  --net_dir NET_DIR     neural network graphs and weights directory. Default
                        is
                        </abs_path_to_scripts/../tensorflow_models/maximask>
  --config_dir CONFIG_DIR
                        configuration file directory. Default is
                        </abs_path_to_script/../data/configs>
  --prior_modif PRIOR_MODIF
                        bool indicating if probability maps should be prior
                        modified. Default is True
  --proba_thresh PROBA_THRESH
                        bool indicating if probability maps should be
                        thresholded. Default is True
  --single_mask SINGLE_MASK
                        bool indicating if resulting masks are joined in a
                        single mask using a binary base. Default is False
  --batch_size BATCH_SIZE
                        neural network batch size. Default is 8. You might
                        want to use a lower value if you have RAM issues
  -v, --verbose         activate output verbosity
```

The raw outputs of MaxiMask are probability maps of the same size than the input image for each contaminant class.

MaxiMask can use 3 different configuration files to respectively select contaminant classes, apply probability prior modification and threshold the probabilities to obtain masks.
By default MaxiMask will prior adjust and threshold the probabilities.

### Configuration files

The 3 configuration files are _classes.flags_, _classes.priors_ and _classes.thresh_. Default versions of those files are located in `maximask_and_maxitrack/data/configs`. **Be careful** that if you want to modify them to use different parameters, you need to use the `--config_dir` option to specify the directory and effectively point to those files.

The adopted syntax consists of two space separated columns:
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

**Be careful** that each configuration file needs to have all the 14 classes, even if some classes are not requested. Also, only the order matters: the abbreviated names are here for convenience when editing the files but are not used to determine the contaminant class.

The abbreviated names stand for:

| Abbreviated name | Full name | Binary Code |
| --- | --- | --- |
| CR | Cosmic Rays | 1 |
| HCL | Hot Columns/Lines | 2 |
| DCL | Dead Columns/Lines/Clusters | 4 |
| HP | Hot Pixels | 8 |
| DP | Dead Pixels | 16 |
| P | Persistence | 32 |
| TRL | TRaiLs | 64 |
| FR | FRinge patterns | 128 |
| NEB | NEBulosities | 256 |
| SAT | SATurated pixels | 512 |
| SP | diffraction SPikes | 1024 |
| OV | OVerscanned pixels | 2048 |
| BBG | Bright BackGround pixel | 4096 |
| BG | Background | 0 |

#### Class selection
Selecting specific classes can be done using _classes.flags_ by indicating 0 or 1 for each class.

#### Probability prior modification
The prior modification aims to modify the MaxiMask output probabilities to match new priors, i.e new class proportions. New prior values can be given in _classes.priors_.

#### Probability thresholding
The probability thresholding aims to threshold the MaxiMask output probabilities to obtain uint8 maps instead of float32 maps. One can use various thresholds to trade off true positive rate vs false positive rate. New threshold values can be given in _classes.thresh_.

#### Single mask
If this option is required, MaxiMask will return only one mask by compiling each requested contaminant class using a binary codegiven in the table above.


# MaxiTrack

MaxiTrack behaves similarly to MaxiMask: it can process images using the same formats (specific HDU, specific image, directory or list file):
```
maxitrack <im_path>
```

## Minimal example
If you run:
```
maxitrack test/test_im.fits.fz
```
You should obtain a file in the current directory named `maxitrack.out` containing the line:
```
test_im.fits.fz 0.0000
```
Where the number corresponding to the image name is the probability that this image is affected by tracking error.
When running again, MaxiTrack will append the new results to `maxitrack.out` if it already exists. 

## General use
Here is full description of MaxiMask. It can be obtained by running `maxitrack -h`:
```
usage: maxitrack [-h] [--net_dir NET_DIR] [--prior PRIOR] [--frac FRAC]
                 [--batch_size BATCH_SIZE] [-v]
                 im_path

MaxiTrack command line parameters:

positional arguments:
  im_path               path to the image(s) to be processed

optional arguments:
  -h, --help            show this help message and exit
  --net_dir NET_DIR     neural network graphs and weights directory. Default
                        is
                        </abs_path_to_script/../tensorflow_models/maxitrack>
  --prior PRIOR         prior value to use. Default is 0.05
  --frac FRAC           value specifying a fraction of all the HDUs to use to
                        speed up processing. Default is 1
  --batch_size BATCH_SIZE
                        neural network batch size. Default is 16. You might
                        want to use a lower value if you have RAM issues
  -v, --verbose         activate output verbosity

```

### Probability prior modification
Similarly to MaxiMask, priors can be specified to adjust the output probabilities to new expected class proportions. As there are only two classes in MaxiTrack (tracking or not tracking), only one prior corresponding to the expected proportion of images affected by tracking errors can be directly specified through the `--prior` option. Default is 0.05

### Fraction option
When giving a FITS file containing N HDUs, MaxiTrack will use the N HDUs by default to compute the output probability for the whole field. In order to run MaxiTrack faster, you can specify a number FRAC<N of HDUs to use to compute the output probability through the `--frac` option. Default is 1.


# LICENSE
Copyright (c) 2018 Maxime Paillassa. 

Both code and model weights are released under MIT license. 

See LICENSE for details.
