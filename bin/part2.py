#!/usr/bin/env python
import warnings
warnings.filterwarnings("ignore")


import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *
from cluster_lib import *

import argparse





def get_dim_reducer(dataset, algo, k=None):
    from sklearn.decomposition import PCA,FastICA

    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    features = X_train.shape[1]
    
    if not k:
        k = features

        
    cache = f"{root}/datasets/{dataset}/dimreducer-{algo}-{k}.pkl"
    if os.path.exists(cache):
        reducer = joblib.load(cache)
        print(f"Read {cache}")
        return reducer
    
    if algo=="pca":
        reducer = PCA(n_components=k)
    elif algo=="ica":
        reducer = FastICA(n_components=k, whiten=True)
    else:
        sys.exit(f"1 Unknown dim red algo '{algo}'")

    timeit(lambda: reducer.fit(X_train), f"{algo} projection calculation for {dataset}, k={k}")
    #reduced_X_train=timeit(lambda: reducer.transform(X_train), f"{algo} projection execution for {dataset}")

    joblib.dump(reducer, cache)
    print(f"Wrote {cache}")
    return reducer

from scipy.stats import kurtosis
def describe_dim_reduction(dataset, algo, k=None):
    X_train_raw, y_train_raw, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    
    features = X_train.shape[1]
    reducer = get_dim_reducer(dataset, algo, k=k)
    X_train_reduced = reducer.transform(X_train)
    print(f"X var:",np.var(X_train, axis=0))
    print("X mean:",np.mean(X_train, axis=0))
    print(f"X kurtosis:",kurtosis(X_train, axis=0))
    print(f"{algo} var:",np.var(X_train_reduced, axis=0))
    print(f"{algo} kurtosis:",kurtosis(X_train_reduced, axis=0))

    
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'part2 util')

    parser.add_argument("-a", "--action", required=True, help="action")
    parser.add_argument("-s", "--dataset", required=False, help="dataset name, eg: ds1)")
    parser.add_argument("-k", required=False, help="specify k")
    parser.add_argument("-d", "--dralgo", required=False, help="dimension reduction algorithm (eg: pca)")
    parser.add_argument("-r", "--reclean", action="store_const", const=True, default=False, help="rerun data clean commands")    
    opts = parser.parse_args()
    dataset = opts.dataset
    k = opts.k
    if k:
        k=int(k)
    if opts.action=="list":
        for ds in spec['datasets']:
            print(f"{ds}:    {spec['datasets'][ds]['name']}")
        exit(0)
    elif opts.action=="make-dim-reducer":
        if not dataset:
            sys.exit(f"Must specify dataset for {opts.action} action with --dataset <ds>")
        if not opts.dralgo:
            sys.exit(f"Must specify dim reduction algo for {opts.action} action with --dralgo <algo>")
        reducer = get_dim_reducer(dataset, opts.dralgo, k=k)
    elif opts.action=="describe-dim-reduction":
        if not dataset:
            sys.exit(f"Must specify dataset for {opts.action} action with --dataset <ds>")
        if not opts.dralgo:
            sys.exit(f"Must specify dim reduction algo for {opts.action} action with --dralgo <algo>")
        describe_dim_reduction(dataset, opts.dralgo, k=k)

        
    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
