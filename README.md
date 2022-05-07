# bnSample
python3 scripts for 
- Reading BIF format to pymc3 objects and generate metadata in json format
- sampling from Bayesian Networks and evaluating divergence between sample and distribution
- computing the probabilities of the sampled combinations, and compute the JS divergence between sample and distribution.

BIF format only supports finite domain variables, and so do scripts in this repository. 
Additionally, size of each variable's domain cannot exceed 127.

In order to use the command line interfaces, you are recommended to including the script directory in PATH, create a separate work directory with a subdirectory "models" and put your BIF files in "models".

## Sampling

    bnSample.py model.bif <output_name.nc|output_csv_dir> sample_size
if output name ends with .nc, sampling results will be saved in netCDF format
else, it is considered a directory to store sampling results in multiple CSV files, one for each chain.
Note: sampleCount.py currently only supports CSV files saved in ./samples/dataset_name/

## Aggregating the sample

    sampleCount.py dataset_name
It will count the sampling results from CSV files in ./samples/dataset_name, and compute log probabilities for each unique value combinations.
