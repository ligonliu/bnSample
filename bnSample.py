#!/usr/bin/python3

import pymc3 as pm, arviz, os, shutil, sys, glob
from BN import BN
from tempfile import mkdtemp


def getMetaData(bn_obj:BN):

    # sample_dir must have CSV in it

    temp_dir = mkdtemp()

    with bn_obj.pymc3 as model:
        db = pm.backends.Text(name=temp_dir, model=model)
        pm.sample(draws=1, trace=db)  # run sample just 1 time

    csv_files = glob.glob(temp_dir+'/*.csv')
    assert len(csv_files) >= 1

    header = open(csv_files[0]).readline().strip().split(',')

    metadata = {'VarNames': bn_obj.getVarNames(),
            'ValLabels': {vn:bn_obj.getValLabels(vn) for vn in bn_obj.getVarNames()},
            'Header': header}

    shutil.rmtree(temp_dir)
    return metadata



if __name__ == '__main__':

    if len(sys.argv)<4:
        print('Usage: bnSample.py model_name output_name.nc sample_size')
        print('sample will be saved in output_name.nc as netCDF file')
        print('actual sampled size may be slightly larger than sample_size')
        exit(0)

    output_file_name = sys.argv[2]

    thread_count = os.cpu_count()

    model_file = "models/{0}.bif".format(sys.argv[1])
    assert os.path.exists(model_file)

    bn_obj = BN(model_file)
    model = bn_obj.pymc3
    sample_size = int(sys.argv[3])
    draws = sample_size // thread_count

    with model as model:
        metadata = bn_obj.getMetaData()
        is_netcdf = (output_file_name[-3:]=='.nc')
        if is_netcdf:
            trace = pm.sample(draws, cores=thread_count, return_inferencedata=True)
            arviz.to_netcdf(trace, output_file_name)
            # it could be read out by
            # v = next(trace.values())
            # df = v.to_dataframe()  , could also to_ndarray and some other formats

        else:
            db = pm.backends.Text(name=output_file_name, model=model)
            pm.sample(draws, trace=db, cores=thread_count, return_inferencedata=False)

