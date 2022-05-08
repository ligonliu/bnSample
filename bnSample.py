#!/usr/bin/env python3

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
        print('Usage: bnSample.py model_name <csv|netCDF> sample_size')
        print('sample will be saved in samples/model_name/ as CSV files or samples/model_name.nc as netCDF file')
        print('actual sampled size may be slightly larger than sample_size')
        print('in which case, discard some initial samples for better convergence')
        exit(0)

    thread_count = os.cpu_count()
    model_name = sys.argv[1]
    model_file = "models/{0}.bif".format(model_name)
    assert os.path.exists(model_file)

    output_format = sys.argv[2]

    bn_obj = BN(model_file)
    model = bn_obj.pymc3
    sample_size = int(sys.argv[-1])
    draws = 1 + sample_size // thread_count

    if not os.path.exists('samples'):
        os.mkdir('samples')

    with model as model:
        metadata = bn_obj.getMetaData()

        if output_format=='netcdf':
            trace = pm.sample(draws, cores=thread_count, return_inferencedata=True)
            output_file_name = 'samples/{0}.nc'.format(model_name)
            arviz.to_netcdf(trace, output_file_name)
            # it could be read out by
            # v = next(trace.values())
            # df = v.to_dataframe()  , could also to_ndarray and some other formats
        elif output_format=='csv':
            db = pm.backends.Text(name='samples/{0}'.format(model_name), model=model)
            trace = pm.sample(draws,trace=db, cores=thread_count, return_inferencedata=True)
            db.close()

