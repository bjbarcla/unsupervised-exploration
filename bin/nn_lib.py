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

    #print(cnf)


    total = sum( [sum(x) for x in cnf])
    correct = shave(cnf[0][0] + cnf[1][1])
    acc = shave(correct / total)
    prec = shave(avg(precisions))
    f1 = shave(avg(fscores))
    rec = shave(avg(recalls))

    #print(accuracy)

    #sys.exit("stop")
    agg=avg([acc, prec, f1, rec])
    return acc, prec, f1, rec, agg


def cook_X(recipe,X):
    if recipe=="Unmodified":
        return X
    else:
        sys.exit("unimplemented recipe provided to cook_X: "+recipe)
        
    
def nn_train_score(dataset, recipe, iter=1):
    X_train, y_train, X_test, y_test =  get_prepared_training_and_test_data(dataset)

    X_train = cook_X(recipe, X_train)
    X_test = cook_X(recipe, X_test)
    
    topology = spec['datasets'][dataset]['mlp_topology']
    mlpid = re.sub('\s+','',re.sub('[\[\](),]+','-',str(topology)) + "-" + recipe)
    pkl = f"datasets/{dataset}/mlp-model-{mlpid}-iter-{iter}.pkl"
    ttime = f"datasets/{dataset}/mlp-model-{mlpid}-iter-{iter}.traintime"
    if os.path.exists(pkl):
        model = joblib.load(pkl)
        print(f"Read {pkl}")
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

    return tracc, trprec, trf1, trrec, tragg,tstacc, tstprec, tstf1, tstrec, tstagg


    
