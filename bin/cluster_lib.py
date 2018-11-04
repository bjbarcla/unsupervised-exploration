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

# def bench_k_means(labels,estimator, name, data):
#     t0 = time()
#     estimator.fit(data)
#     sample_size = 300
#     print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
#           % (name, (time() - t0), estimator.inertia_,
#              metrics.homogeneity_score(labels, estimator.labels_),
#              metrics.completeness_score(labels, estimator.labels_),
#              metrics.v_measure_score(labels, estimator.labels_),
#              metrics.adjusted_rand_score(labels, estimator.labels_),
#              metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
#              metrics.silhouette_score(data, estimator.labels_,
#                                       metric='euclidean',
#                                       sample_size=sample_size)))


# def kmeans_graph_lda(dataset):
#     # do all the same data prep as we did in the analysis project
#     X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
#     k = spec['datasets'][dataset]['best-k-kmeans']

#     #from sklearn.lda import LDA
#     from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA



#     kmeans_model = timeit(lambda: KMeans(n_clusters=k, random_state=1).fit(X_train), f"kmeans {dataset} k={k}")
#     labels = np.array(kmeans_model.labels_)
#     print(labels)


    
#     #    # Plot all three series
#     #    plt.scatter(lda_transformed[y==1][0], lda_transformed[y==1][1], label='Class 1', c='red')
#     #    plt.scatter(lda_transformed[y==2][0], lda_transformed[y==2][1], label='Class 2', c='blue')
#     #    plt.scatter(lda_transformed[y==3][0], lda_transformed[y==3][1], label='Class 3', c='lightgreen')

#     ##lda_transformed = pd.DataFrame(lda.fit_transform(X_train, labels)) # https://www.apnorton.com/blog/2016/12/19/Visualizing-Multidimensional-Data-in-Python/
#     lda = LDA(n_components=2) #2-dimensional LDA
#     lda_2d = timeit(lambda: lda.fit_transform(X_train, labels), f"LDA projection for {dataset}")
#     print(lda_2d.shape)

#     # https://www.dummies.com/programming/big-data/data-science/how-to-visualize-the-clusters-in-a-k-means-unsupervised-learning-model/
#     import pylab as pl


#     markers = "o,x,+,*,1,2,3,4,s,p,h,d,v".split(",") # https://matplotlib.org/api/markers_api.html
#     colors = ['r', 'g', 'y', 'c', 'm', 'b', 'k'] # https://matplotlib.org/users/colors.html  # no w

#     limit=2000
#     for i in range(0, min(limit, lda_2d.shape[0])):
#         # 13 markers, 7 colors.  relatively prime, so repeats well after 50, our max k
#         label = labels[i]
#         color = colors[label % len(colors)]
#         marker = markers[label % len(markers)]

#         pl.scatter(lda_2d[i,0],lda_2d[i,1],c=color, marker=marker, alpha=0.3)
        
#     pl.title(f'K-means clusters for dataset {dataset}')
#     #pl.show()
#     png=f"{dataset}-kmeans-clusterviz.png"
#     pl.savefig(png, bbox_inches='tight')
#     print("Wrote "+png)



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
    plt.title(f"{algo} {measure} score\nvarying k and covariance type")
    plt.xlabel("k")
    plt.ylabel("Score")
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


    
    #metrics.homogeneity_score(y_train, labels),
    #metrics.completeness_score(y_train, labels),
    #metrics.v_measure_score(y_train, labels),
    #metrics.adjusted_rand_score(y_train, labels),
    #metrics.adjusted_mutual_info_score(y_train,  labels),

    
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








from time import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

np.random.seed(42)

#digits = load_digits()
#data = scale(digits.data)

#n_samples, n_features = data.shape
#n_digits = len(np.unique(digits.target))
#labels = digits.target

#sample_size = 300

#print("n_digits: %d, \t n_samples %d, \t n_features %d"
#      % (n_digits, n_samples, n_features))


#print(82 * '_')
#print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')


def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    sample_size=3000
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

def bench_gmm(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    sample_size=3000
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), 
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))


def kmeans_bench(dataset, k):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    #for k in range

    #for k in [2,6,10,25,35,40]:
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X_train)
    bench_k_means(kmeans_model,f"k-means k={k}", X_train, y_train)
    centroids = kmeans_model.cluster_centers_
    #print(centroids)

def gmm_bench(dataset, k, cov_type):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    #for k in range

    #for k in [2,6,10,25,35,40]:
    from sklearn.mixture import GaussianMixture
    clusterer = timeit(lambda: GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0), f"gmm k={k} cov_type={cov_type}")
    bench_gmm(clusterer,f"gmm k={k}", X_train, y_train)
    centroids = kmeans_model.cluster_centers_
    #print(centroids)


def plot_kmeans_bench_maybe(dataset, k):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    #for k in range

    #for k in [2,6,10,25,35,40]:
    kmeans_model = KMeans(n_clusters=k, random_state=1).fit(X_train)
    bench_k_means(kmeans_model,f"k-means k={k}", X_train, y_train)


#    # in this case the seeding of the centers is deterministic, hence we run the
#    # kmeans algorithm only once with n_init=1
#    pca = PCA(n_components=k, whiten=True).fit(data)
#    bench_k_means(KMeans(init=pca.components_, n_clusters=n_digits, n_init=1),
#                  name="PCA-based",
#                  data=data)
#    print(82 * '_')

    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=2).fit_transform(X_train)
    kmeans = KMeans(init='k-means++', n_clusters=k, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title(f'K-means clustering on PCA-reduced {dataset} dataset k={k}\n'
              f'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()
