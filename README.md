# FEL-UQ
SLAC LCLS UQ Project


## Details: 
This repo contains quantile regression uncertainty quantification results for the SLAC LCLS FEL pulse energy prediction. The data used for this project consists of SLAC LCLS archive data, where a single sample consists of 76 scalar values as inputs (upstream of the pulse energy detector), and a final scalar output value (photon pulse energy at a gas detector). 

This project is in progress. Notebooks with Bayesian neural network training will be added shortly.


This work has been submitted as a workshop paper at New in ML, NeurIPS 2020 (awaiting acceptance/rejection.)

## The Models:
All models currently in this repo are for uncertainty quantification using quantile regression neural networks. The base model simply uses all of the data to make a models to predict the median measured value, and a 2.5% quantile and 97.5% quantile prediction. Interpolation models (models trained on a subset of data then used to predict on another, time-ordered subset) are trained and evaluated in Interp and Interp2.

## Authors: 
Primary author (to contact with questions) - Lipi Gupta (lipigupta at uchicago.edu)

SLAC Researchers: Aashwin Mishra and Auralee Edelen

## Requirements:
Required packages are listed in the `environment.yml` file. 


It is suggested to use the bash script to make the environment. Simply run the script:

```./prepare.sh ```

``` conda activate feluq ```


This repo uses git Large File Storage (LFS) so installation of git-lfs may be needed. 
[For git lfs information](https://git-lfs.github.com/).

You can create a unique conda environment for this repo by doing the following (manually): 

```conda env create -f environment.yml```

```conda activate feluq```

This method may require manual installation of scikit-learn from pip, due to some yet-unsolved bug.
