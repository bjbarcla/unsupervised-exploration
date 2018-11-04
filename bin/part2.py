#!/usr/bin/env python


import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *
from cluster_lib import *
from dimension_reduction_lib import *


import argparse




    

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
    elif opts.action=="ica-kurtcurve":
        ica_kurtcurve(dataset)
    elif opts.action=="pca-eigenplot":
        pca_eigenplot(dataset)
    elif opts.action=="lda-evrplot":
        lda_explained_variance_ratios(dataset)
    elif opts.action=="pca-loss":
        pca_loss(dataset,k)
    elif opts.action=="rp-loss":
        rp_loss(dataset,k)
    elif opts.action=="loss":
        reducer_loss(dataset,opts.dralgo,k)
    elif opts.action=="lossplot":
        reducer_loss_plot(dataset)
    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
