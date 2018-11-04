#!/usr/bin/env python


import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *
from cluster_lib import *
from dimension_reduction_lib import *
from nn_lib import *

import argparse




    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'part4 util')

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
    if opts.action=="report":
        iter = 1
        header = '"Dataset","Xval Accuracy","Xval Precision","Xval F1 Score","Xval Recall","Xval Aggregate"'
        print(header)
        for dataset in ["ds1","ds4"]:
            for recipe in "Unmodified".split(","):
                tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg = nn_train_score(dataset, recipe, iter=iter)
                row = f"{dataset} ({recipe})" + ",".join( [f"{x:.3f}" for x in [tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg]] )
                print(row)

    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
