#!/usr/bin/env python3
import numpy as np
import scipy.spatial


def JSDivergences(counts, probs):

    # read through logprob.txt.gz, store counts and probs in lists
    sum_probs = np.sum(probs)

    raw_jsd = scipy.spatial.distance.jensenshannon(counts, probs)
    real_jsd=raw_jsd

    if (sum_probs<1):
        probs.resize(probs.shape[0]+1, refcheck=False)
        probs[-1]= 1.0-sum_probs
        counts.resize(counts.shape[0]+1, refcheck=False)
        counts[-1] = 0
        real_jsd = scipy.spatial.distance.jensenshannon(counts, probs)

    return raw_jsd, real_jsd, sum_probs


if __name__ == '__main__':
    import sys
    if len(sys.argv)<2:
        print("Usage: JSDivergences.py model_name")
        exit(0)
    model_name = sys.argv[1]
    counts = np.load("samples/{0}/proj/all.count.npy".format(model_name))
    probs = np.load("samples/{0}/proj/all.prob.npy".format(model_name))

    raw_jsd, real_jsd, sum_probs = JSDivergences(counts,probs)
    print("RawJSD, RealJSD, SampleCoverage = {0} , {1} , {2}".format(raw_jsd,real_jsd,sum_probs))