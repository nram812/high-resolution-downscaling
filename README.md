# High-resolution downscaling with Interpretable Deep Learning

## Python Environment Installation 
There shouldn't be too many challenges in getting this environment up to speed, and a conda environment file will be added shortly. 
```bash
conda create -n high_res_env -python=3.8.5
```
While this should work with any Python environment up till 3.10, we recommend using Python 3.8.5. 

The Python packages can be installed using pip or conda. Here is a list of required packages using pip
```bash
pip install tensorflow
pip install xarray 
pip install matplotlib
pip install pandas
pip install numpy
pip install cartopy
pip install dask
```

## Data
There are two key data files used. Firstly the large-scale reanalysis fields (X). These have been extracted from ERA5 and reinterpolated to a spatial resolution of 0.5 degrees of New Zealand. 

ERA5 data can be found here: https://unsw-my.sharepoint.com/:u:/g/personal/z5458529_ad_unsw_edu_au/EQb9kFaO1bxHuvSWa_2dOXMBpo93h9ma2Cm1PgPPn5GEZQ

MSWEP precipitation data can be found here: https://unsw-my.sharepoint.com/:u:/g/personal/z5458529_ad_unsw_edu_au/EUXpUOnSUT1MhOFEq0JfBiABD1lxPj-XOB5UG90FYFZoqA

Please email neelesh.rampal@niwa.co.nz if you need access to the data. 

While precipitation from the VCSN dataset was used in training, we have provided MSWEP daily precipitation data instead (y). Here a model is trained to map from X->y. 

The MSWEP data is daily from 1982-2018 (different to the data used in the paper). The ERA5 data is from 1974-2020 and has been regridded using CDO to a resolution of 0.5 degrees. 

## Scripts and Notebooks
We've provided a basic implementation of the code to begin with. This notebook (in the notebook folder), benchmarks three CNN approaches. The notebook contains an overview of how to train deep learning models. 

1. CNN with a GAMMA loss function
2. Linear Dense with a MSE loss function
3. Linear Dense with GAMMA loss function

Further experiments will be added in due course. 

Please note that to load all the models you will need to use the src.py files. 

