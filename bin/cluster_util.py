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


actions_list="list,kmeans,kmeans-graph,kmeans-plot-ksweep,gmm-sweepk,kmeans-plot-clusters,gmm-plot-ksweep,gmm-sweepk-CH,gmm-plot-ksweep-CH,kmeans-sweepk-CH,kmeans-plot-ksweep-CH"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'clustering util')

    actions = actions_list.split(",")
    parser.add_argument("-a", "--action", required=True, choices=actions, help="action")
    parser.add_argument("-s", "--dataset", required=False, help="dataset name, eg: ds1)")
    parser.add_argument("-r", "--reclean", action="store_const", const=True, default=False, help="rerun data clean commands")    
    opts = parser.parse_args()
    dataset = opts.dataset
    if opts.action=="list":
        for ds in spec['datasets']:
            print(f"{ds}:    {spec['datasets'][ds]['name']}")
        exit(0)
    elif opts.action=="kmeans":
        # do all the same data prep as we did in the analysis project
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        # http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        from sklearn import metrics
        from sklearn.cluster import KMeans

        ks = spec['datasets'][dataset]['k-sweep']

        for k in range(*ks):

            kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
            labels = kmeans_model.labels_
            sil_score = timeit(lambda: metrics.silhouette_score(X_train, labels, metric='euclidean'), f"silscore {dataset} k={k}")
            print(f"Dataset={dataset} k={k} sil={sil_score}")

    elif opts.action=="kmeans-sweepk-CH":
        # do all the same data prep as we did in the analysis project
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        # http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        from sklearn import metrics
        from sklearn.cluster import KMeans

        ks = spec['datasets'][dataset]['k-sweep']

        for k in range(*ks):

            kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
            labels = kmeans_model.labels_
            sil_score = timeit(lambda: metrics.calinski_harabaz_score(X_train, labels), f"CH {dataset} k={k}")
            print(f"Dataset={dataset} k={k} CH={sil_score}")

    elif opts.action=="gmm-sweepk":
        from sklearn.mixture import GaussianMixture
        from sklearn import metrics
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        #k=2
        ks = spec['datasets'][dataset]['k-sweep']
        #cov_type in ['spherical', 'diag', 'tied', 'full']
        #cov_type="full"
        for k in range(*ks):
            for cov_type in cov_types:
                clusterer = timeit(lambda: GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0), f"gmm k={k} cov_type={cov_type}")
                clusterer.fit(X_train)
                cluster_labels=clusterer.predict(X_train)
                sil_score = timeit(lambda: metrics.silhouette_score(X_train, cluster_labels), f"sil gmm k={k} cov_type={cov_type}")
                print(f"Dataset={dataset} k={k} cov_type={cov_type} sil={sil_score}")
                #print(f"GMM: cov_type={cov_type} k={k} sil={sil}")
    elif opts.action=="gmm-sweepk-CH": # http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        from sklearn.mixture import GaussianMixture
        from sklearn import metrics
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        #k=2
        ks = spec['datasets'][dataset]['k-sweep']
        #cov_type in ['spherical', 'diag', 'tied', 'full']
        #cov_type="full"
        for k in range(*ks):
            for cov_type in cov_types:
                clusterer = timeit(lambda: GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0), f"gmm k={k} cov_type={cov_type}")
                clusterer.fit(X_train)
                cluster_labels=clusterer.predict(X_train)
                #Compute the Calinski and Harabaz score.
                #It is also known as the Variance Ratio Criterion.
                #The score is defined as ratio between the within-cluster dispersion and the between-cluster dispersion.
                CH_score = timeit(lambda: metrics.calinski_harabaz_score(X_train, cluster_labels), f"CH gmm k={k} cov_type={cov_type}")
                print(f"Dataset={dataset} k={k} cov_type={cov_type} CH={CH_score}")
                #print(f"GMM: cov_type={cov_type} k={k} sil={sil}")
        
    elif opts.action=="kmeans-graph":
        kmeans_graph(opts.dataset)

    elif opts.action=="gmm-plot-ksweep":
        gmm_plot_sweep(dataset, "silhouette")
        exit(0)        
    elif opts.action=="gmm-plot-ksweep-CH":
        gmm_plot_sweep(dataset, "CH")
        exit(0)        
    elif opts.action=="kmeans-plot-ksweep-CH":
        kmeans_plot_sweep(dataset, "CH")
        exit(0)        
    elif opts.action=="kmeans-plot-ksweep":
        kmeans_plot_sweep(dataset, "silhouette")
        exit(0)
    elif opts.action=="kmeans-plot-clusters":
        kmeans_graph(dataset)
        exit(0)
    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
