import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *


    
    
def get_dim_reducer(dataset, algo, k=None, reset=False):
    from sklearn.decomposition import PCA,FastICA
    from sklearn import random_projection
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.manifold import TSNE
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    features = X_train.shape[1]
    
    if not k:
        k = features

    cache = f"{root}/datasets/{dataset}/dimreducer-{algo}-{k}.pkl"
        
    if cache and os.path.exists(cache) and not reset:
        reducer = joblib.load(cache)
        print(f"Read {cache}")
        return reducer
    
    if algo=="pca":
        reducer = PCA(n_components=k, whiten=True)
    elif algo=="ica":
        reducer = FastICA(n_components=k, whiten=True)
    elif algo=="rp":
        reducer = random_projection.SparseRandomProjection(n_components=k)
    elif algo=="lda":
        reducer = LDA(n_components=k)
    elif algo=="tsne":
        reducer = TSNE(n_components=k)
    else:
        sys.exit(f"1 Unknown dim red algo '{algo}'")

    if algo=="lda":
        timeit(lambda: reducer.fit(X_train, y_train), f"{algo} projection calculation for {dataset}, k={k}")
    else:
        timeit(lambda: reducer.fit(X_train), f"{algo} projection calculation for {dataset}, k={k}")
    #reduced_X_train=timeit(lambda: reducer.transform(X_train), f"{algo} projection execution for {dataset}")

    if cache:
        joblib.dump(reducer, cache)
        
    print(f"Wrote {cache}")
    return reducer

from scipy.stats import kurtosis
def describe_dim_reduction(dataset, algo, k=None):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    
    features = X_train.shape[1]
    n_samples = X_train.shape[0]
    reducer = get_dim_reducer(dataset, algo, k=k)

    X_train_reduced = reducer.transform(X_train)

    if algo=="pca":
        # https://stackoverflow.com/questions/31909945/obtain-eigen-values-and-vectors-from-sklearn-pca#31941631
        #cov_matrix = np.dot(X_train.T, X_train) / n_samples
        for idx, evec_eval in enumerate(  zip(reducer.components_, reducer.explained_variance_ )    ):
            eigenvector, eigenvalue = evec_eval
            print(f"PC{idx}, eigenvector={eigenvector} eigenvalue={eigenvalue}")
        print(f"{algo} var:",np.var(X_train_reduced, axis=0))
    elif algo=="ica":
        kurts = kurtosis(X_train_reduced, axis=0)
        avgkurts = np.mean([abs(x) for x in kurts])
        print(f"{algo} avgabs kurtosis={avgkurts} ; kurtoses:",kurts)    
    else:
        sys.exit(f"ddr - unimp {algo}")

def pca_loss(dataset, k):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    from numpy.testing import assert_array_almost_equal

    # https://stackoverflow.com/questions/36566844/pca-projecting-and-reconstruction-in-scikit#36567821
    features = X_train.shape[1]
    reducer = get_dim_reducer(dataset, "pca", k=k)
    
    X_train_pca = reducer.transform(X_train)
    X_train_pca2 = (X_train - reducer.mean_).dot(reducer.components_.T)
    #assert_array_almost_equal(X_train_pca, X_train_pca2)
    
    #X_train_pca2 = (X_train - reducer.mean_).dot(reducer.components_.T)
    X_projected = reducer.inverse_transform(X_train_pca)
    #X_projected2 = X_train_pca.dot(reducer.components_) + reducer.mean_
    #assert_array_almost_equal(X_projected, X_projected2)
    
    loss = ((X_train - X_projected) ** 2).mean()
    print(k,loss)
    #reducer_loss(dataset, reducer)

def reducer_loss(dataset, algo, k):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    reducer = get_dim_reducer(dataset, algo, k=k)
    X_train_reduced = reducer.transform(X_train)
    X_projected = reducer.inverse_transform(X_train_reduced)
    #X_projected = X_train_reduced.dot(reducer.components_) #+ reducer.mean_
    loss = ((X_train - X_projected) ** 2).mean()
    print(loss)


    
def reducer_loss_plot(dataset):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    algos = ['ica','pca']

    features = X_train.shape[1]

    ks = []
    losses = dict()

    for algo in algos:
        losses[algo] = []


    for k in range(2,features+1):
        ks.append(k)
        for algo in algos:
            reducer = get_dim_reducer(dataset, algo, k=k)

            X_train_reduced = reducer.transform(X_train)

            if algo=="ica":
                #https://stackoverflow.com/questions/35407669/independent-component-analysis-ica-in-python
                mm = reducer.mixing_  # Get estimated mixing matrix
                X_projected = np.dot(X_train_reduced, mm.T) + reducer.mean_
            else:
                X_projected = reducer.inverse_transform(X_train_reduced)

                

            
            loss = ((X_train - X_projected) ** 2).mean()
            losses[algo].append(loss)

        
    plt.title(f"Dataset {dataset}: Reconstruction loss for {algo}")

    linetypes = ['x-', 'o-', '*-', '+-']
    
    for idx, algo in enumerate(algos):
        plt.plot(ks,losses[algo], linetypes[idx], label=f"{algo}")
        
    plt.xlabel("components")
    plt.ylabel(f"loss (MSE)")
    plt.legend(loc="best")
    
    png=f"{dataset}-loss-curve.png"
    plt.savefig(png, bbox_inches='tight')

    print("Wrote "+png)
#    csv=f"{dataset}-{algo}-loss-curve.csv"
#    with open(csv,"w") as fh:
#        for k,d in zip(ks,losses):
#            fh.write(f"{k},{d}\n")
#    print("Wrote "+csv)


def pca_eigenplot(dataset):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    
    features = X_train.shape[1]

    pcs = list(range(1,features+1))
    reducer = get_dim_reducer(dataset, "pca")
    eigs=reducer.explained_variance_

    plt.title(f"Dataset {dataset}: PCA Component Eigenvalues")
    plt.plot(pcs,eigs, 'x-', label=f"eigenvalue (explained variance)")
    plt.xlabel("principal component")
    plt.ylabel(f"eigenvalue (explained variance)")
    png=f"{dataset}-pca-eigenplot.png"
    plt.savefig(png, bbox_inches='tight')
    print("Wrote "+png)
               
    csv=f"{dataset}-pca-eigenvals.csv"
    with open(csv,"w") as fh:
        for k,d in zip(pcs,eigs):
            fh.write(f"{k},{d}\n")
    print("Wrote "+csv)

    
        
def ica_kurtcurve(dataset):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    
    features = X_train.shape[1]

    ks = []
    avgkurts = []
    kurtses = []
    for k in range(2,features+1):
        ks.append(k)

        reducer = get_dim_reducer(dataset, "ica", k=k)

        X_train_reduced = reducer.transform(X_train)

        kurts = kurtosis(X_train_reduced, axis=0)
        kurtses.append(kurts)
        avgkurts.append(np.mean([abs(x) for x in kurts]))

    plt.title(f"Dataset {dataset}: Average absolute kurtosis of ICA components")
    plt.plot(ks,avgkurts, 'x-', label=f"average absolute kurtosis")
    plt.xlabel("components")
    plt.ylabel(f"average absolute kurtosis")
    png=f"{dataset}-ica-avgabskurtosos-ksweep.png"
    csv=f"{dataset}-ica-avgabskurtosos-ksweep.csv"
    
    plt.savefig(png, bbox_inches='tight')
    print("Wrote "+png)

    with open(csv,"w") as fh:
        for k,d in zip(ks,avgkurts):
            fh.write(f"{k},{d}\n")
    print("Wrote "+csv)

    csv=f"{dataset}-ica-kurtoses-ksweep.csv"
    with open(csv,"w") as fh:
        for k,d in zip(ks,kurtses):
            s=",".join([f"{x}" for x in d])
            fh.write(f"{k},{s}\n")
    print("Wrote "+csv)
    plt.clf()

    #print(f"X var:",np.var(X_train, axis=0))
    #print("X mean:",np.mean(X_train, axis=0))
    #print(f"X kurtosis:",kurtosis(X_train, axis=0))

    

    
    
