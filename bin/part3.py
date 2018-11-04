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
from cluster_p3_lib import *
from dimension_reduction_lib import *

import argparse






def get_reducer_X_transformer(dataset,reducer,n_components):
    red = get_dim_reducer(dataset, reducer, n_components)

    transformer = lambda X: red.transform(X)
    
    reduction=f"{reducer} with n_components={n_components}"
    return transformer, reduction

#actions_list="list,kmeans-sweepk,kmeans-graph,kmeans-plot-ksweep,gmm-sweepk,kmeans-plot-clusters,gmm-plot-ksweep,gmm-sweepk-CH,gmm-plot-ksweep-CH,kmeans-sweepk-CH,kmeans-plot-ksweep-CH,gmm-graph"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'clustering util')

    #actions = actions_list.split(",")
    parser.add_argument("-a", "--action", required=True)#, choices=actions, help="action")
    parser.add_argument("-n", "--ncomp",  help="n_components for feedin clusterer")#, choices=actions, help="action")
    parser.add_argument("-d", "--reducer", help="Dimensionality reducer spec")#, choices=actions, help="action")
    parser.add_argument("-s", "--dataset", required=False, help="dataset name, eg: ds1)")
    parser.add_argument("-r", "--reclean", action="store_const", const=True, default=False, help="rerun data clean commands")
    opts = parser.parse_args()
    dataset = opts.dataset
    #transformer, reduction = get_reducer_X_transformer(opts.reducer) 
    if opts.action=="list":
        for ds in spec['datasets']:
            print(f"{ds}:    {spec['datasets'][ds]['name']}")
        exit(0)

    elif opts.action=="graphs":
        for reducer in ["pca","ica","rp","lda"]:
            for dataset in ["ds1","ds4"]:
                ncomps = spec['datasets'][dataset][f"best-{reducer}-ncomponents"]
                for ncomp in ncomps:
                    for algo in ["kmeans","gmm"]:
                        logfile=f"part3logs/{algo}-sweepk-{dataset}-{reducer}-{ncomp}-sil.log"
                        png=f"part3res/{dataset}-{algo}-{reducer}-{ncomp}-silhouette-ksweep.png"
                        measure="silhouette"
                        clust_plot_sweep_p3(dataset, measure, logfile, algo, png,reducer,ncomp)

                        logfile=f"part3logs/{algo}-sweepk-{dataset}-{reducer}-{ncomp}-CH.log"
                        png=f"part3res/{dataset}-{algo}-{reducer}-{ncomp}-CH-ksweep.png"
                        measure="CH"
                        clust_plot_sweep_p3(dataset, measure, logfile, algo, png,reducer,ncomp)
                        
                        
                        
                        
                        

                        

                        #{algo}-sweepk-{dataset}-{reducer}-{ncomp}-sil.log"
                        


    elif opts.action=="jobs":
        jf="part3-all.jobs"
        with open(jf,"w") as fh:
            for reducer in ["pca","ica","rp","lda"]:
                for dataset in ["ds1","ds4"]:
                    #print( spec['datasets'][dataset] )
                    ncomps = spec['datasets'][dataset][f"best-{reducer}-ncomponents"]
                    for ncomp in ncomps:
                        for algo in ["gmm","kmeans"]:
                            logfile=f"part3logs/{algo}-sweepk-{dataset}-{reducer}-{ncomp}-sil.log"
                            fh.write(f"launcher.sh --log-file {logfile} bin/part3.py -a {algo}-sweepk --ncomp {ncomp} -s {dataset} --reducer {reducer}\n")
                            logfile=f"part3logs/{algo}-sweepk-{dataset}-{reducer}-{ncomp}-CH.log"
                            fh.write(f"launcher.sh --log-file {logfile} bin/part3.py -a {algo}-sweepk-CH --ncomp {ncomp} -s {dataset} --reducer {reducer}\n")
        print(f"Wrote {jf}")
        
    elif opts.action=="kmeans-sweepk":
        # do all the same data prep as we did in the analysis project
        ncomp=int(opts.ncomp)
        reducer=opts.reducer
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)
        # http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        from sklearn import metrics
        from sklearn.cluster import KMeans

        
        ks = spec['datasets'][dataset]['k-sweep']

        for k in range(*ks):
            kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} reduced by {reduction} clust_k={k}")
            labels = kmeans_model.labels_
            sil_score = timeit(lambda: metrics.silhouette_score(X_train, labels, metric='euclidean'), f"silscore {dataset} k={k}")
            print(f"Dataset={dataset} k={k} sil={sil_score}")

    elif opts.action=="kmeans-sweepk-CH":
        # do all the same data prep as we did in the analysis project
        ncomp=int(opts.ncomp)
        reducer=opts.reducer
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)

        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)
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
        
        #X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        ncomp=int(opts.ncomp)
        reducer=opts.reducer
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)

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
    elif opts.action=="gmm-sweepk-ics":
        from sklearn.mixture import GaussianMixture
        from sklearn import metrics
        #X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
        ncomp=int(opts.ncomp)
        reducer=opts.reducer
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)
        
        #k=2
        ks = spec['datasets'][dataset]['k-sweep']
        #cov_type in ['spherical', 'diag', 'tied', 'full']
        #cov_type="full"
        ks_plot = []
        aics = dict()
        bics = dict()

        for ct in cov_types:
            aics[ct] = []
            bics[ct] = []
        
        for k in range(*ks):
            ks_plot.append(k)
            for cov_type in cov_types:
                pkl=f"datasets/{dataset}/gmm-clusterer-{k}.pkl"
                if os.path.exists(pkl):
                    clusterer=joblib.load(pkl)
                else:
                    clusterer = timeit(lambda: GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0), f"gmm k={k} cov_type={cov_type}")
                    clusterer.fit(X_train)
                    joblib.dump(clusterer,pkl)
                    
                cluster_labels=clusterer.predict(X_train)
                aic_score = clusterer.aic(X_train)
                aics[cov_type].append(aic_score)
                bic_score = clusterer.bic(X_train)
                bics[cov_type].append(bic_score)
                print(f"Dataset={dataset} k={k} cov_type={cov_type} bic={bic_score}")
                print(f"Dataset={dataset} k={k} cov_type={cov_type} aic={aic_score}")
                #print(f"GMM: cov_type={cov_type} k={k} sil={sil}")
        algo="gmm"
        plt.title(f"{algo} aic and bic scores\nvarying k and covariance type\n(lower is better)")
        ltsaic = ["rx-","g*-","bo-","k+-"] #line types
        ltsbic = ["rx","g*","bo","k+"] #line types
        for idx in range(0,4):
            cov_type = cov_types[idx]
            ltaic = ltsaic[idx]
            ltbic = ltsbic[idx]
            plt.plot(ks_plot,aics[cov_type], ltaic, label=f"aic,cov_type")
            plt.plot(ks_plot,bics[cov_type], ltbic, label=f"bic,cov_type")

        plt.xlabel("k")
        plt.ylabel("Score")
        plt.legend(loc="best")
        png=f"{dataset}-{algo}-aicbic-ksweep.png"
        plt.savefig(png, bbox_inches='tight')
        print("Wrote "+png)
        plt.clf()

        
        
    elif opts.action=="gmm-sweepk-CH": # http://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation
        from sklearn.mixture import GaussianMixture
        from sklearn import metrics

        ncomp=int(opts.ncomp)
        reducer=opts.reducer
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)
        X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)

        #X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
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
    elif opts.action=="gmm-graph":
        gmm_graph(opts.dataset)

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
    elif opts.action=="kmeans-bench":
        k = spec['datasets'][dataset]['best-k-kmeans']
        kmeans_bench(dataset, k)
    elif opts.action=="gmm-bench":
        k = spec['datasets'][dataset]['best-k-gmm']
        cov_type = spec['datasets'][dataset]['best-cov-type']
        gmm_bench(dataset, k, cov_type)
    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
