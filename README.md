# bnSample
python3 scripts for 
- Reading BIF format to pymc3 objects and generate metadata in json format
- sampling from Bayesian Networks and evaluating divergence between sample and distribution
- computing the probabilities of the sampled combinations, and compute the JS divergence between sample and distribution.

BIF format only supports finite domain variables, and so do scripts in this repository. 
Additionally, size of each variable's domain cannot exceed 127.

Please also read the jupyter notebook for usages.

## Setup

Scripts in this project strictly depend on earlier package versions. Latest versions of the same packages does NOT work.

### Conda environment setup 

    conda create -n pymc3 python=3.8.13 pip
    conda activate pymc3
    pip install arviz==0.11.0 pyagrum==0.18.0 pymc3==3.9.3

In order to use the command line interfaces, you are recommended to including the script directory in PATH, create a separate work directory with a subdirectory "models" and put your BIF files in "models".

## Sampling

    bnSample.py model_name <netCDF|csv> sample_size
if output name ends with .nc, sampling results will be saved in netCDF format
else, it is considered a directory to store sampling results in multiple CSV files, one for each chain.
Note: sampleCount.py currently only supports CSV files saved in ./samples/dataset_name/

## Aggregating the sample

    sampleCount.py model_name

It will count the sampling results from CSV files in ./samples/dataset_name, and compute probabilities for each unique value combination. The results will be stored in ./samples/dataset_name/proj

## computing JS Divergence

    JSDivergences.py model_name
    
It will output the raw and real JS divergence between sample and distribution, as well as sample coverage (defined by sum of probabilities of sampled random value combinations).
