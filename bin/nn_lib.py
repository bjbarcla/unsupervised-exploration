import sys
import re
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import time
from dimension_reduction_lib import *

def evaluate_model_data(model, X, y):
    y_pred = model.predict(X)
    cnf_mat = confusion_matrix(y, y_pred)
    precisions, recalls, fscores, support = precision_recall_fscore_support(y, y_pred, average=None)
    precision, recall, fscore, support = precision_recall_fscore_support(y, y_pred, average='micro')

    return {'precisions':precisions,
            'recalls':recalls,
            'fscores':fscores,
            'precision':precisions,
            'recall':recalls,
            'fscore':fscores,
            'confusion_matrix':cnf_mat,
            'y_labels': [str(i) for i in list(set(y)|set(y_pred))] }

def avg(x):
    return sum(x)/len(x)

def shave(x):
    return int(x * 1000)/1000

def get_perf_stats(model, X, y):
    yhat = model.predict(X)
    precisions, recalls, fscores, support = precision_recall_fscore_support(y, yhat, average=None)
    cnf = confusion_matrix(y, yhat)

    total = sum( [sum(x) for x in cnf])
    correct = shave(cnf[0][0] + cnf[1][1])
    acc = shave(correct / total)
    prec = shave(avg(precisions))
    f1 = shave(avg(fscores))
    rec = shave(avg(recalls))
    agg=avg([acc, prec, f1, rec])
    return acc, prec, f1, rec, agg


        
def get_reducer_X_transformer(dataset,reducer,n_components):
    red = get_dim_reducer(dataset, reducer, n_components)

    transformer = lambda X: red.transform(X)
    
    reduction=f"{reducer} with n_components={n_components}"
    return transformer, reduction


def onehot_cluster_features(X,clusterer):
    labels = clusterer.predict(X)
    new_feature_categorical = np.reshape(labels, (len(labels),1))
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(new_feature_categorical)
    new_features_onehot = enc.transform(new_feature_categorical)
    return new_features_onehot
    
def augment_X_labels(X, clusterer):
    new_features_onehot = onehot_cluster_features(X,clusterer).toarray()
    #print(X.shape)
    #print(type(X))
    #print(new_features_onehot.shape)
    #print(type(new_features_onehot))
    final = np.hstack((X,new_features_onehot))
    #print(final.shape)
    return final

def replace_X_labels(X, clusterer):
    new_features_onehot = onehot_cluster_features(X,clusterer)
    return new_features_onehot



def get_cluster_X_transformer(dataset, clusteralgo, augment_or_replace):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)
    
    if clusteralgo=="gmm":
        from sklearn.mixture import GaussianMixture
        k = spec['datasets'][dataset]['best-k-gmm']
        cov_type = spec['datasets'][dataset]['best-cov-type']
        clusterer = GaussianMixture(n_components=k, covariance_type=cov_type, max_iter=200, random_state=0) #, f"gmm k={k} cov_type={cov_type}")
        reduction=f"{augment_or_replace} features with {clusteralgo} (k={k},cov_type={cov_type})"
    elif clusteralgo=="kmeans":
        from sklearn.cluster import KMeans
        k = spec['datasets'][dataset]['best-k-kmeans']
        reduction=f"{augment_or_replace} features with {clusteralgo} (k={k})"
        clusterer = KMeans(n_clusters=k, random_state=1) #, f"kmeans {dataset} k={k}")
    else:
        raise(ValueError(f"illegal cluster algo [{clusteralgo}]"))
    timeit(lambda: clusterer.fit(X_train), f"{clusteralgo} fit for {augment_or_replace} transformer for dataset {dataset}")

    #X_train_labels=clusterer.predict(X_train)
    #X_test_labels =clusterer.predict(X_test)
    #print( X_train_labels.head )

    if augment_or_replace == "augment":
        transformer = lambda X: augment_X_labels(X, clusterer)
    elif augment_or_replace == "replace":
        transformer = lambda X: replace_X_labels(X, clusterer)
    else:
        raise(ValueError(f"Illegal value for augment_or_replace: [{augment_or_replace}]"))

    return transformer,reduction


    
def nn_train_score(dataset, recipe, iter=1):

    mcl = re.match('(kmeans|gmm)_(augment|replace)', recipe)
    m4=re.match('([^_]+)_(\d+)d', recipe)
    mdos=re.match('([^_]+)_(\d+)d_(kmeans|gmm)_(augment|replace)', recipe)
    
    if mcl:
        clusteralgo=mcl.group(1)
        augment_or_replace = mcl.group(2)
        transformer, reduction = get_cluster_X_transformer(dataset, clusteralgo, augment_or_replace)
    elif m4:
        reducer = m4.group(1)
        ncomp = int(m4.group(2))
        transformer, reduction = get_reducer_X_transformer(dataset,reducer,ncomp)
        #print(f"HELLO {reduction}")
        #sys.exit("stop")
    elif mdos: #porque no los dos?
        reducer = m4.group(1)
        ncomp = int(m4.group(2))
        clusteralgo=mcl.group(3)
        augment_or_replace = mcl.group(4)
        transformer1, reduction1 = get_cluster_X_transformer(dataset, clusteralgo, augment_or_replace)
        transformer2, reduction2 = get_reducer_X_transformer(dataset,reducer,ncomp)
        reduction = f"{reduction1} then {reduction2}"
        transformer = lambda x: transformer2(transformer1(x))
        
    elif recipe=="Unmodified":
        transformer = lambda x: x
        reduction = "Unmodified"
    else:
        raise(ValueError(f"Unhandled recipe [{recipe}]"))
        
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset, x_transformer=transformer)
    
    topology = spec['datasets'][dataset]['mlp_topology']
    mlpid = re.sub('\s+','',re.sub('[\[\](),]+','-',str(topology)) + "-" + recipe)
    pkl = f"datasets/{dataset}/mlp-model-{mlpid}-iter-{iter}.pkl"
    ttime = f"datasets/{dataset}/mlp-model-{mlpid}-iter-{iter}.traintime"
    if os.path.exists(pkl):
        model = joblib.load(pkl)
        print(f"Read {pkl}")
        fh = open(ttime)
        delta = float(fh.read())
        fh.close()
        print(f"Read {ttime}")
    else:
        maxiter=500
        if dataset=="ds1":
            model = MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto', beta_1=0.9,
                                  beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                  #hidden_layer_sizes=(100, 100, 100),
                                  hidden_layer_sizes=topology,
                                  learning_rate='constant',
                                  learning_rate_init=0.001, max_iter=maxiter, momentum=0.9,
                                  nesterovs_momentum=True, power_t=0.5,
                                  random_state=42, shuffle=True, solver='adam', tol=0.0001,
                                  validation_fraction=0.1, verbose=True, warm_start=False)
        elif dataset=="ds4":
            model = MLPClassifier(activation='tanh', alpha=0.05623413251903491, batch_size='auto',
                                  beta_1=0.9, beta_2=0.999, early_stopping=False, epsilon=1e-08,
                                  #hidden_layer_sizes=(40, 40, 10),
                                  hidden_layer_sizes=topology,
                                  learning_rate='constant',
                                  learning_rate_init=0.001, max_iter=maxiter, momentum=0.9,
                                   nesterovs_momentum=True, power_t=0.5,
                                  random_state=42, shuffle=True, solver='adam', tol=0.0001,
                                  validation_fraction=0.1, verbose=True, warm_start=False)
        else:
            sys.exit("No such dataset: "+dataset)

        t0 = time.time()
        model.fit(X_train,y_train)
        delta = time.time() - t0
        with open(ttime,"w") as fh:
            fh.write(str(delta))
        joblib.dump(model,pkl)
        print(f"Wrote {pkl}")

        
    tracc, trprec, trf1, trrec, tragg = get_perf_stats(model, X_train, y_train)
    tstacc, tstprec, tstf1, tstrec, tstagg = get_perf_stats(model, X_test, y_test)

    return tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg, delta


    
