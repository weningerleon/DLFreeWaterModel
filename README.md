# Overview

In diffusion MRI acquisitions, the acquired signal is integrated over rather big voxels. If this voxel contains free-water as well as white matter, the signal can't be directly used to compare the microstructure to voxels containing no white matter. Thus, for voxels close to CSF or in presence of edema, the tissue-only signal should be extracted before microstructure analysis or white matter tractography.

Here, this free-water elimination is carried out with individually extracted synthetic data.
For an exemple of usage, see the jupyter notebook "example.ipynb".

Details on the method and implementations can be found in the MICCAI CDMRI workshop [paper](https://www.lfb.rwth-aachen.de/bibtexupload/pdf/WEN19b.pdf).

## Installation

The code has been tested on Ubuntu 18.04, python 3.6 and pyTorch 1.0.  
The packages dipy (0.16), numpy (1.16) and scipy (1.2) are necessary. 

## Citation

If you use our work, please cite the following paper:

```tex
@inproceedings{WEN19b,
title = {Free-Water Correction in Diffusion MRI: A Reliable and Robust Learning Approach},
author = {Leon Weninger and Simon Koppers and Chuh-Hyoun Na and Kerstin Juetten and Dorit Merhof}}
year = {2019},
booktitle = {MICCAI Workshop on Computational Diffusion MRI (CDMRI)},
```

