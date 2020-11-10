# FEL-UQ
SLAC LCLS UQ Project


## Details: 
This repo contains quantile regression uncertainty quantification results for the SLAC LCLS FEL pulse energy prediction. This project is in progress. This work has been submitted as a workshop paper at New in ML, NeurIPS 2020 (awaiting acceptance/rejection.)

## Authors: 
Primary author (to contact with questions) - Lipi Gupta <lipigupta at uchicago.edu>

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
