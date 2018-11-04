import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale


from time import time

def clust_plot_sweep_p3(dataset, measure, logfile, algo, png, reducer,ncomp):
    print(f"INFO dataset,reducer,ncomp,measure,algo,bestk,bestscore")
    if algo=="kmeans":
        kmeans_plot_sweep_p3(dataset, measure, logfile, algo, png, reducer,ncomp)
    elif algo=="gmm":            
        gmm_plot_sweep_p3(dataset, measure, logfile, algo, png, reducer,ncomp)
    else:
        raise(ValueError(f"bad algo [{algo}]"))


def gmm_plot_sweep_p3(dataset, measure, logfile, algo, png, reducer,ncomp):
    #sys.exit(f"not yet [{algo}]")
    algo="gmm"
    if measure=="silhouette":
        #logfile = f"{algo}-sweepk-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) cov_type=(\\S+) sil=(\S+)"
    elif measure=="CH":
        #logfile = f"{algo}-sweepk-CH-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) cov_type=(\\S+) CH=(\S+)"
    else:
        system.exit(f"huh? measure={measure}")
        
    ks=[]
    sils=dict()
    cov_types=["tied"]
    for cov_type in cov_types:
        sils[cov_type] = []

    floatpatt="\\S+"
    
    
    #print(patt)
    #exit(1)
    with open(logfile) as fh:
        for line in fh:
            line = re.sub('/nfs/pdx/disks/icf_env_disk001/bjbarcla/gwa/issues/ml/mlai/lib/python3.6/site-packages/sklearn/ensemble/weight_boosting.py:29:','',line)
            m=re.match(patt, line)
            if m:
                cov_type = m.group(2)
                k = int(m.group(1))
                #print(f"matched k={k} covtype={cov_type}")
                if not k in ks:
                    ks.append(k)
                if cov_type=="tied":
                    sils[cov_type].append(float(m.group(3)))
    #        else:
    #            print(line)
    #print(sils)
    #exit(1)
    #for cov_type in cov_types:
    #    print( cov_type, len(sils[cov_type]))
    #exit(1)
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots()
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    bestk=0
    bestsil=0
    bestcovtype=0
    for idx, k in enumerate(ks):
        for cov_type in cov_types:
            #print(idx,k,cov_type)
            sil = sils[cov_type][idx]
            if sil > bestsil:
                bestk = k
                bestcovtype = cov_type
                bestsil = sil

    lts = ["rx-","g*-","bo-","k+-"] #line types
    for idx in range(0,len(cov_types)):
        cov_type = cov_types[idx]
        lt = lts[idx]
        plt.plot(ks,sils[cov_type], lt, label=cov_type)
    title=f"{dataset} reduced by {reducer} n_components={ncomp}\n{algo} {measure} score\nvarying k and covariance type\nbest k={bestk} ({measure} score={bestsil:.3f})"
    plt.title(title)
    print(f"INFO {dataset},{reducer},{ncomp},{measure},{algo},{bestk},{bestsil:.3f}")
    plt.xlabel("k")
    plt.ylabel("Score")
    plt.legend(loc="best")
    #png=f"{dataset}-{algo}-{measure}-ksweep.png"
    plt.savefig(png, bbox_inches='tight')
    #print("Wrote "+png)
    plt.clf()


def kmeans_plot_sweep_p3(dataset, measure, logfile, algo, png, reducer,ncomp):
    #algo="kmeans"
    if measure=="silhouette":
#        logfile = f"{algo}-sweepk-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) sil=(\S+)"
    elif measure=="CH":
#        logfile = f"{algo}-sweepk-CH-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) CH=(\S+)"
    else:
        system.exit(f"huh? measure={measure}")


    #logfile = f"kmeans-sweepk-{opts.dataset}.log"
    ks=[]
    sils=[]
    #floatpatt="\\d+\\.\\d+"
    with open(logfile) as fh:
        for line in fh:
            #m=re.match(f"Dataset={opts.dataset} k=(\d+) sil=({floatpatt})", line)
            m=re.match(patt, line)
            if m:
                ks.append(int(m.group(1)))
                sils.append(float(m.group(2)))
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter
    fig, ax = plt.subplots()
    #ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    bestk=0
    bestsil=0
    for k,sil in zip(ks,sils):
        if sil > bestsil:
            bestk=k
            bestsil=sil


    #print(ks)
    #print(sils)
    plt.plot(ks,sils, 'x-', label=f"{measure} score")
    plt.title(f"{dataset} reduced by {reducer} n_components={ncomp}\n{algo} {measure} Score, varying k\nbest k={bestk} ({measure} score={bestsil:.3f})")#\nbest k={bestk} ({measure} score={bestsil:.3f})")
    print(f"INFO {dataset},{reducer},{ncomp},{measure},{algo},{bestk},{bestsil:.3f}")
    plt.xlabel("k")
    plt.ylabel(f"{measure} Score")
    #png=f"{dataset}-{algo}-{measure}-ksweep.png"
    plt.savefig(png, bbox_inches='tight')
    #print("Wrote "+png)
    plt.clf()
