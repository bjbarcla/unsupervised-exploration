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
    parser = argparse.ArgumentParser(description = 'part5 util')

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
    elif opts.action=="foo":
        xf = get_cluster_X_transformer(dataset, "kmeans", "augment")
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        new = xf(X_test)
        print(X_test.shape)
        print(new.shape)
    elif opts.action=="losdos":
        iter = 1
        rep = ""
        header = '"Dataset","Xval Accuracy","Xval Precision","Xval F1 Score","Xval Recall","Xval Aggregate","Test Set Accuracy","Test Set Precision","Test Set F1 Score","Test Set Recall","Test Set Aggregate","Train Time"'
        rep += header + "\n"
        print(header)
        best_reducer_recipes = dict()
        best_reducer_recipes['ds1'] = ["pca_10d","rp_10d"]
        best_reducer_recipes['ds4'] = ["pca_30d","pca_55d","rp_30d","rp_57d"]

        best_cluster_recipes = dict()
        best_cluster_recipes['ds1'] = ["kmeans_augment","gmm_augment"]
        best_cluster_recipes['ds4'] = ["kmeans_augment","gmm_augment"]

        for dataset in ["ds4","ds1"]:
            recipes = ["Unmodified"]

            for redrec in best_reducer_recipes[dataset]:
                for clurec in best_cluster_recipes[dataset]:
                    recipes.append( f"{redrec}_{clurec}" )
            for recipe in recipes:
                tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg, traintime = nn_train_score(dataset, recipe, iter=iter)
                row = f'"{dataset} ({recipe})",' + ",".join( [f"{x:.3f}" for x in [tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg]] ) + f",{traintime}"
                print(row)
                rep += row + "\n"
        repfile="part5_losdos_report.csv"
        with open(repfile,"w") as fh:
            fh.write(rep)
        print(f"Write {repfile}")
                    

        
    elif opts.action=="report":
        iter = 1
        rep = ""
        header = '"Dataset","Xval Accuracy","Xval Precision","Xval F1 Score","Xval Recall","Xval Aggregate","Test Set Accuracy","Test Set Precision","Test Set F1 Score","Test Set Recall","Test Set Aggregate","Train Time"'
        rep += header + "\n"
        print(header)
        
        for dataset in ["ds4","ds1"]:
            recipes = ["Unmodified"]
            for clu in ['kmeans','gmm']:
                for method in ['replace','augment']:
                    recipes.append(f"{clu}_{method}")
            for recipe in recipes:
                tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg, traintime = nn_train_score(dataset, recipe, iter=iter)
                row = f'"{dataset} ({recipe})",' + ",".join( [f"{x:.3f}" for x in [tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg]] ) + f",{traintime}"
                print(row)
                rep += row + "\n"
        repfile="part5_report.csv"
        with open(repfile,"w") as fh:
            fh.write(rep)
        print(f"Write {repfile}")
                
