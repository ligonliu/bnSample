#!/usr/bin/env python3

import csv, glob, gc, os, pyAgrum as gum, numpy as np
from itertools import chain
from collections import Counter
from BN import BN


def getRowIterator(csv_file):
    fp = csv.reader(open(csv_file))
    next(fp)
    return map(lambda x: tuple(map(int, x)),fp)

def countInCSVs(csv_files):
    iters = [getRowIterator(c) for c in csv_files]
    chain_iter = chain(*iters)
    counts = Counter(chain_iter)
    return counts

def getP(bn_obj:BN, header:list, vals):

    insta:gum.Instantiation = bn_obj.agrum.completeInstantiation()

    assert len(header)==len(vals)

    # vals could be numbers, in which case, we need to transform to strings

    for i in range(0,len(header)):
        if isinstance(vals[i], int):
            insta.chgVal(header[i], bn_obj.getValLabels(header[i])[vals[i]])
        else:
            insta.chgVal(header[i], vals[i])

    return bn_obj.agrum.jointProbability(insta)


if __name__ == '__main__':

    import sys,gzip

    if len(sys.argv)<2:
        print('Usage: sampleCount.py dataset_name [dataset2_name] ...')
        exit(0)

    for n in range(1, len(sys.argv)):

        gc.collect()
        ds_name = sys.argv[n]

        sample_csv_dir = 'samples/' + ds_name
        assert os.path.exists(sample_csv_dir)
        csv_files = glob.glob(sample_csv_dir + '/*.csv')
        assert len(csv_files) > 0

        bn_obj = BN('models/{0}.bif'.format(ds_name))

        header = open(csv_files[0]).readline().strip().split(',')

        print('Counting {0}....'.format(ds_name))

        count = countInCSVs(csv_files)

        sys.stdout.write('{0} uniques, sorting by frequency....'.format(len(count)))

        sorted_by_frequency = count.most_common()
        del count
        gc.collect()

        print('Sort Complete')

        # dt_string = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        count_save_dir = sample_csv_dir

        # os.mkdir(count_save_dir)

        print('Writing to count.txt.gz')
        fp_out = gzip.open(count_save_dir+'/count.txt.gz', 'wt')

        for k,v in sorted_by_frequency:
            fp_out.write(repr(k))
            fp_out.write(',')
            fp_out.write(repr(v))
            fp_out.write('\n')

        fp_out.close()

        # generate the proj binary files
        proj_dir = count_save_dir + '/proj'
        os.mkdir(proj_dir)
        all_data = np.zeros(shape=(len(sorted_by_frequency), len(header)), dtype='int8')
        counts = np.zeros(shape=(len(sorted_by_frequency),), dtype='int64')
        probs = np.zeros(shape=(len(sorted_by_frequency),), dtype='float64')

        i=0

        for k,v in sorted_by_frequency:
            counts[i] = v
            all_data[i] = k
            probs[i] = getP(bn_obj,header, k)
            i+=1

        assert i==len(sorted_by_frequency)

        np.save(proj_dir+'/all.npy', all_data)
        np.save(proj_dir+'/all.count.npy', counts)
        np.save(proj_dir+'/all.prob.npy', probs)

        # shutil.copyfile(sample_csv_dir + '/metadata.json', count_save_dir + '/metadata.json')

        if os.path.exists(count_save_dir+'/count.txt.gz') and os.path.exists(count_save_dir+'/metadata.json'):
            print('Counting and metadata saved in {0}'.format(count_save_dir))
        else:
            print('Counting file or metadata generation failed in {0}'.format(count_save_dir))
