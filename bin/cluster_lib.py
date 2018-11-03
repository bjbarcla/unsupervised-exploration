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

def bench_k_means(labels,estimator, name, data):
    t0 = time()
    estimator.fit(data)
    sample_size = 300
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))
    

def kmeans_graph_lda(dataset):
    # do all the same data prep as we did in the analysis project
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    k = spec['datasets'][dataset]['best-k-kmeans']

    #from sklearn.lda import LDA
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



    kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
    labels = np.array(kmeans_model.labels_)
    print(labels)


    
    #    # Plot all three series
    #    plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Class 1', c='red')
    #    plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
    #    plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

    ##lda_transformed = pd.DataFrame(lda.fit_transform(X_train, labels)) # https://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/
    lda = LDA(n_components=2) #2-dimensional LDA
    lda_2d = timeit(lambda: lda.fit_transform(X_train, labels), f"LDA projection for {dataset}")
    print(lda_2d.shape)

    # https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/
    import pylab as pl


    markers = "o,x,+,*,1,2,3,4,s,p,h,d,v".split(",") # https://matplotlib.org/api/markers_api.html
    colors = ['r', 'g', 'y', 'c', 'm', 'b', 'k'] # https://matplotlib.org/users/colors.html  # no w

    limit=2000
    for i in range(0, min(limit, lda_2d.shape[0])):
        # 13 markers, 7 colors.  relatively prime, so repeats well after 50, our max k
        label = labels[i]
        color = colors[label % len(colors)]
        marker = markers[label % len(markers)]

        pl.scatter(lda_2d[i,0],lda_2d[i,1],c=color, marker=marker, alpha=0.3)
        
    pl.title(f'K-means clusters for dataset {dataset}')
    #pl.show()
    png=f"{dataset}-kmeans-clusterviz.png"
    pl.savefig(png, bbox_inches='tight')
    print("Wrote "+png)



cov_types = ['spherical', 'diag', 'tied', 'full']



def kmeans_plot_sweep(dataset, measure):
    algo="kmeans"
    if measure=="silhouette":
        logfile = f"{algo}-sweepk-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) sil=(\S+)"
    elif measure=="CH":
        logfile = f"{algo}-sweepk-CH-{dataset}.log"
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
    plt.title(f"{algo} {measure} Score, varying k\nbest k={bestk} ({measure} score={bestsil:.3f})")
    plt.xlabel("k")
    plt.ylabel(f"{measure} Score")
    png=f"{dataset}-{algo}-{measure}-ksweep.png"
    plt.savefig(png, bbox_inches='tight')
    print("Wrote "+png)
    plt.clf()


def gmm_plot_sweep(dataset, measure):
    algo="gmm"
    if measure=="silhouette":
        logfile = f"{algo}-sweepk-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) cov_type=(\\S+) sil=(\S+)"
    elif measure=="CH":
        logfile = f"{algo}-sweepk-CH-{dataset}.log"
        patt=f"Dataset={dataset} k=(\\d+) cov_type=(\\S+) CH=(\S+)"
    else:
        system.exit(f"huh? measure={measure}")
        
    ks=[]
    sils=dict()
    for cov_type in cov_types:
        sils[cov_type] = []

    floatpatt="\\S+"
    
    
    print(patt)
    #exit(1)
    with open(logfile) as fh:
        for line in fh:
            m=re.match(patt, line)
            if m:
                cov_type = m.group(2)
                k = int(m.group(1))
                print(f"matched k={k} covtype={cov_type}")
                if not k in ks:
                    ks.append(k)
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
            print(idx,k,cov_type)
            sil = sils[cov_type][idx]
            if sil > bestsil:
                bestk = k
                bestcovtype = cov_type
                bestsil = sil

    lts = ["rx-","g*-","bo-","k+-"] #line types
    for idx in range(0,4):
        cov_type = cov_types[idx]
        lt = lts[idx]
        plt.plot(ks,sils[cov_type], lt, label=cov_type)
    plt.title(f"{algo} {measure} score\nvarying k and covariance type\nbest silhouette score={bestsil:.3f} @ cov_type={bestcovtype},k={bestk}")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    plt.legend(loc="best")
    png=f"{dataset}-{algo}-{measure}-ksweep.png"
    plt.savefig(png, bbox_inches='tight')
    print("Wrote "+png)
    plt.clf()



def kmeans_graph_pca(dataset):
    # do all the same data prep as we did in the analysis project
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    k = spec['datasets'][dataset]['best-k-kmeans']

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    #oldname#transformed = timeit(lambda: pd.DataFrame(pca.fit_transform(X_train)), f"PCA projection for {dataset}")
    pca_2d = timeit(lambda: pca.fit_transform(X_train), f"PCA projection for {dataset}")
    print(pca_2d.shape)
    kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
    labels = kmeans_model.labels_
    print(labels)
    # https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/
    import pylab as pl

    markers = "o,x,+,*,1,2,3,4,s,p,h,d,v".split(",") # https://matplotlib.org/api/markers_api.html
    colors = ['r', 'g', 'y', 'c', 'm', 'b', 'k'] # https://matplotlib.org/users/colors.html  # no w

    limit=3000
    for i in range(0, min(limit, pca_2d.shape[0])):
        label = labels[i]

        # 13 markers, 7 colors.  relatively prime, so repeats well after 50, our max k
        color = colors[label % len(colors)]
        marker = markers[label % len(markers)]

        pl.scatter(pca_2d[i,0],pca_2d[i,1],c=color, marker=marker, alpha=0.6)
        
    pl.title(f'K-means clusters for dataset {dataset}')
    #pl.show()
    png=f"{dataset}-kmeans-clusterviz.png"
    pl.savefig(png, bbox_inches='tight')
    print("Wrote "+png)


#kmeans_graph = kmeans_graph_lda

def kmeans_graph(dataset):
    # do all the same data prep as we did in the analysis project
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    which = spec['datasets'][dataset]['kmeans-graph']
    if which == "lda":
        kmeans_graph_lda(dataset)
    else:
        kmeans_graph_pca(dataset)


# in:
#  - dimension reduction method
#  - reduced component count
#  - training data
#
# # reduce data, do training, measure score. (don't care about learning curves...)
# 
# out:
#  - xval score
#  - test score


def gmm_graph(dataset):
    # do all the same data prep as we did in the analysis project
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    k = spec['datasets'][dataset]['best-k-gmm']
    cov_type = spec['datasets'][dataset]['best-cov-type']

    from sklearn.decomposition import PCA
    from sklearn.mixture import GaussianMixture
    from sklearn import metrics

    pca = PCA(n_components=2)
    #oldname#transformed = timeit(lambda: pd.DataFrame(pca.fit_transform(X_train)), f"PCA projection for {dataset}")
    pca_2d = timeit(lambda: pca.fit_transform(X_train), f"PCA projection for {dataset}")
    print(pca_2d.shape)

    clusterer = timeit(lambda: GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0), f"gmm k={k} cov_type={cov_type}")
    timeit(lambda: clusterer.fit(X_train), f"gmm fit")
    labels=timeit(lambda: clusterer.predict(X_train), "gmm run")

    #kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
    #labels = kmeans_model.labels_
    #print(labels)
    # https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/
    import pylab as pl

    markers = "o,x,+,*,1,2,3,4,s,p,h,d,v".split(",") # https://matplotlib.org/api/markers_api.html
    colors = ['r', 'g', 'y', 'c', 'm', 'b', 'k'] # https://matplotlib.org/users/colors.html  # no w

    limit=3000
    for i in range(0, min(limit, pca_2d.shape[0])):
        label = labels[i]

        # 13 markers, 7 colors.  relatively prime, so repeats well after 50, our max k
        color = colors[label % len(colors)]
        marker = markers[label % len(markers)]

        pl.scatter(pca_2d[i,0],pca_2d[i,1],c=color, marker=marker, alpha=0.6)
        
    pl.title(f'GMM clusters for dataset {dataset}')
    #pl.show()
    png=f"{dataset}-gmm-clusterviz.png"
    pl.savefig(png, bbox_inches='tight')
    print("Wrote "+png)
