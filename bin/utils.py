import os
import sys
from os.path import dirname,abspath
import yaml
import pandas as pd
from sklearn.externals import joblib
import numpy as np
import re
import pathlib
import itertools
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
sys.path.append(dirname(abspath(__file__)))


root = dirname(dirname(__file__))

from misc import *
from sklearn import tree
#from sklearn import neural_network
from sklearn import ensemble
from sklearn import svm
from sklearn import neighbors

#from imblearn.pipeline import make_pipeline
#from imblearn.under_sampling import EditedNearestNeighbours, RepeatedEditedNearestNeighbours

import time
def human_duration(seconds, granularity=2):
    intervals = (
        ('weeks', 604800),  # 60 * 60 * 24 * 7
        ('days', 86400),    # 60 * 60 * 24
        ('hours', 3600),    # 60 * 60
        ('minutes', 60),
        ('seconds', 1),
        )
    result = []

    for name, count in intervals:
        value = seconds // count
        if value:
            seconds -= value * count
            if value == 1:
                name = name.rstrip('s')
            result.append("{} {}".format(value, name))
    if len(result[:granularity]) < 1:
        return "less than 1 second"
    else:
        return ', '.join(result[:granularity])

def timeit(thunk, prefix="unnamed task"):
    start=time.time()
    rv = thunk()
    end=time.time()
    elapsed = (end - start)
    duration = human_duration(elapsed)
    print( f"** -> {prefix} elapsed seconds {elapsed} <- took {duration}" )
    return rv


def get_spec():
    specfile = f"{root}/spec.yml"
    with open(specfile) as yf:
        spec = yaml.load(yf)
    return spec

spec = get_spec()

def classifiers_list():
    return list(spec['classifiers'].keys())

def datasets_list():
    return [x for x in spec['datasets'] if ((not 'verdict' in spec['datasets'][x]) or spec['datasets'][x]['verdict'] != 'reject')]

def scoring_list():
    return spec['scoring_methods']


def all_ds_clf_pairs():
    a = datasets_list()
    b = classifiers_list()
    return list(itertools.product(a, b))
    
def all_ds_clf_scoring_tuples():
    a = datasets_list()
    b = classifiers_list()
    c = scoring_list()
    return list(itertools.product(a, b, c))


def backticks(cmd):
    return os.popen(cmd).read()

#def dataset_csv_path(dataset):
#    csvfile = spec['datasets']['ds1']['csv']
#    csv_path = f"{root}/datasets/{dataset}/{csvfile}"
#    return csv_path

def get_dataset_dir(dataset):
    return f"{root}/datasets/{dataset}"


def get_results_dir(dataset):
    if not dataset:
        dest = f"{root}/results"
    else:
        dest = f"{root}/results/{dataset}"

    if not os.path.exists(dest):
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
    return dest
    

def sanitize_filename(fname):
    return re.sub("[^a-zA-Z_\-+/0-9]","_", fname)

def mkresultsdirs():
    subdirs = "complexity-curves,confusion-matrices,iter-curves,learning-curves,report-data".split(",")
    if not os.path.exists(f"{root}/results"):
        os.mkdir(f"{root}/results")
    for subdir in subdirs:
        targ = f"{root}/results/{subdir}"
        
        if not os.path.exists(targ):
            os.mkdir(targ)


def get_searcher(dataset, algo, scoring):
    results_dir = f"{root}/results"
    pickle = f"{results_dir}/{dataset}.{algo}.{scoring}.cvsearch.pkl"
    if not os.path.exists(pickle):
        raise(ValueError(f"Error: file does not exist: {pickle}"))
    return joblib.load(pickle)
    
def get_bestfit_model(dataset, algo, scoring):
    results_dir = f"{root}/results"
    pickle = f"{results_dir}/{dataset}.{algo}.{scoring}.best_model.pkl"
    if not os.path.exists(pickle):
        raise(ValueError(f"Error: file does not exist: {pickle}"))
    return joblib.load(pickle)

def get_hyperparam_dict(clf,algo):
    hp_names = list(sorted(spec['distribs'][algo]['param_distributions'].keys()))
    rv = {hp_name: getattr(clf, hp_name) for hp_name in hp_names}
    return rv

def get_bestfit_params(dataset, algo, scoring):
    best_model = get_bestfit_model(dataset, algo, scoring)
    best_params = get_hyperparam_dict(best_model, algo)
    return best_params

def get_bestfit_model_untrained(dataset, algo, scoring, hyperparams=None):
    clf_proc_no_hps_code = spec['classifiers'][algo]
    if not hyperparams:
        hyperparams = get_bestfit_params(dataset, algo, scoring)

    clf_best_hps_code = re.sub("\(\)$", "(**hyperparams)", re.sub("\((.+)\)$", "(\\1,**hyperparams)",clf_proc_no_hps_code))
    
    clf = eval( clf_best_hps_code )
    return clf
    
def get_dataset(dataset, reclean=False):
    dest = get_dataset_dir(dataset)
    if not os.path.exists(dest):
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

    orig_pwd = os.environ['PWD']
    os.chdir(dest)

    getfiles = get_ds_prop(dataset, 'getfiles')
    cleancmds= get_ds_prop(dataset, 'cleancmds')
    
    datafile = get_ds_prop(dataset, 'datafile')
    

    
    # download files
    files_to_download = None
    if cleancmds:
        files_to_download = getfiles
    else:
        files_to_download = [datafile]
        
    if not all([os.path.exists(x) for x in files_to_download]):
        for cmd in getcmds:
            print(f"Running cmd: {cmd}")
            rv = os.system(cmd)
            if not rv==0:
                print(f"Fatal: while acquiring dataset {dataset}, Command failed: {cmd}")
                os.chdir(orig_pwd)
                return False
    if not all([os.path.exists(x) for x in files_to_download]):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
        
    # clean files
    if cleancmds:
        if reclean or not os.path.exists(datafile):
            for cmd in cleancmds:
                print(f"Running cmd: {cmd}")
                rv = os.system(cmd)
                if not rv==0:
                    print(f"Fatal: while cleaning dataset {dataset}, Command failed: {cmd}")
                    os.chdir(orig_pwd)
                    return False
    os.chdir(orig_pwd)
    return True
#    print(f"BB> cleancmds={cleancmds}")
#    exit(1)
    
    if not os.path.exists(datafile):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
    os.chdir(orig_pwd)
    return False

def ppwd():
    pwd=os.environ["PWD"]
    print(f"pwd = {pwd}")

def get_ds_prop(dataset, prop, default=None):
    if not dataset in spec['datasets']:
        return default
    if not prop in spec['datasets'][dataset]:
        return default
    return spec['datasets'][dataset][prop]
    
    
def get_dataset_dataframe(dataset, reclean=False):
    if not get_dataset(dataset, reclean=reclean):
        return False
    format =   spec['datasets'][dataset]['format']
    datafile_fullpath = root + "/datasets/" + dataset + "/" + spec['datasets'][dataset]['datafile']

    print(f"Load datafile {datafile_fullpath}")
    if format == 'csv':
        return pd.read_csv(datafile_fullpath)
    elif format == "xls":
        return pd.read_excel(datafile_fullpath,
                             sheet_name=get_ds_prop(dataset, 'sheet'),
                             indexcol=get_ds_prop(dataset, 'indexcol'),
                             skiprows=get_ds_prop(dataset, 'skiprows'))
    else:
        print(f"Abort: [{format}] format not supported for dataset {dataset}")
        return False

def get_dataset(dataset, reclean=False):
    dest = get_dataset_dir(dataset)
    if not os.path.exists(dest):
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

    orig_pwd = os.environ['PWD']
    os.chdir(dest)

    getfiles = get_ds_prop(dataset, 'getfiles')
    cleancmds= get_ds_prop(dataset, 'cleancmds')
    
    datafile = get_ds_prop(dataset, 'datafile')
    

    
    # download files
    files_to_download = None
    if cleancmds:
        files_to_download = getfiles
    else:
        files_to_download = [datafile]
        
    if not all([os.path.exists(x) for x in files_to_download]):
        for cmd in getcmds:
            print(f"Running cmd: {cmd}")
            rv = os.system(cmd)
            if not rv==0:
                print(f"Fatal: while acquiring dataset {dataset}, Command failed: {cmd}")
                os.chdir(orig_pwd)
                return False
    if not all([os.path.exists(x) for x in files_to_download]):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
        
    # clean files
    if cleancmds:
        if reclean or not os.path.exists(datafile):
            for cmd in cleancmds:
                print(f"Running cmd: {cmd}")
                rv = os.system(cmd)
                if not rv==0:
                    print(f"Fatal: while cleaning dataset {dataset}, Command failed: {cmd}")
                    os.chdir(orig_pwd)
                    return False
    os.chdir(orig_pwd)
    return True
#    print(f"BB> cleancmds={cleancmds}")
#    exit(1)
    
    if not os.path.exists(datafile):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
    os.chdir(orig_pwd)
    return False

def ppwd():
    pwd=os.environ["PWD"]
    print(f"pwd = {pwd}")

def get_ds_prop(dataset, prop, default=None):
    if not dataset in spec['datasets']:
        return default
    if not prop in spec['datasets'][dataset]:
        return default
    return spec['datasets'][dataset][prop]

def make_distrib(distrib_type, distrib_params):
    if distrib_type == 'list':
        return distrib_params
    elif distrib_type == 'tuple_list':
        return list([tuple(x) for x in distrib_params])
    elif distrib_type == 'logspace':
        return np.logspace(*distrib_params)
    elif distrib_type == 'arange':
        return np.arange(*distrib_params)
    else:
        None


def get_known_algos():
    return list(spec['distribs'].keys())

def get_hyperparam_option(algo,option):
    if not algo in spec['distribs']:
        raise(ValueError(f"algorithm [{algo}] is not defined in spec.yml in distribs section."))
    
    if not 'options' in spec['distribs'][algo]:
        raise(ValueError(f"algorithm [{algo}] does not define options in spec.yml in distribs section."))

    if not option in spec['distribs'][algo]['options']:
        raise(ValueError(f"algorithm [{algo}] does not define options.{options} in spec.yml in distribs section."))

    return spec['distribs'][algo]['options'][option]


def get_hyperparam_distrib(algo):
    if not algo in spec['distribs']:
        raise(ValueError(f"algorithm [{algo}] is not defined in spec.yml in distribs section."))

    if not 'param_distributions' in spec['distribs'][algo]:
        raise(ValueError(f"algorithm [{algo}] does not define param_distributions in spec.yml in distribs section."))

    algo_info = spec['distribs'][algo]['param_distributions']

    rv = dict()
    for hyperparam in algo_info:
        searchspace = algo_info[hyperparam]
        distrib_type = searchspace[0]
        distrib_params = searchspace[1:]
        dist = make_distrib(distrib_type, distrib_params)
        rv[hyperparam] = dist
    return rv

def get_dataset_dataframe(dataset, reclean=False):
    if not get_dataset(dataset, reclean=reclean):
        return False
    format =   spec['datasets'][dataset]['format']
    datafile_fullpath = root + "/datasets/" + dataset + "/" + spec['datasets'][dataset]['datafile']

    print(f"Load datafile {datafile_fullpath}")
    if format == 'csv':
        return pd.read_csv(datafile_fullpath)
    elif format == "xls":
        return pd.read_excel(datafile_fullpath,
                             sheet_name=get_ds_prop(dataset, 'sheet'),
                             indexcol=get_ds_prop(dataset, 'indexcol'),
                             skiprows=get_ds_prop(dataset, 'skiprows'))
    else:
        print(f"Abort: [{format}] format not supported for dataset {dataset}")
        return False


def get_dataset(dataset, reclean=False):
    dest = get_dataset_dir(dataset)
    if not os.path.exists(dest):
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)

    orig_pwd = os.environ['PWD']
    os.chdir(dest)

    getfiles = get_ds_prop(dataset, 'getfiles')
    cleancmds= get_ds_prop(dataset, 'cleancmds')
    getcmds  = get_ds_prop(dataset, 'getcmds')
    datafile = get_ds_prop(dataset, 'datafile')
    

    
    # download files
    files_to_download = None
    if cleancmds:
        files_to_download = getfiles
    else:
        files_to_download = [datafile]
        
    if not all([os.path.exists(x) for x in files_to_download]):
        for cmd in getcmds:
            print(f"Running cmd: {cmd}")
            rv = os.system(cmd)
            if not rv==0:
                print(f"Fatal: while acquiring dataset {dataset}, Command failed: {cmd}")
                os.chdir(orig_pwd)
                return False
    if not all([os.path.exists(x) for x in files_to_download]):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
        
    # clean files
    if cleancmds:
        if reclean or not os.path.exists(datafile):
            for cmd in cleancmds:
                print(f"Running cmd: {cmd}")
                rv = os.system(cmd)
                if not rv==0:
                    print(f"Fatal: while cleaning dataset {dataset}, Command failed: {cmd}")
                    os.chdir(orig_pwd)
                    return False
    return True
#    print(f"BB> cleancmds={cleancmds}")
#    exit(1)
    
    if not os.path.exists(datafile):
        print("Get dataset failed for {dataset} ")
        os.chdir(orig_pwd)
        return True
    os.chdir(orig_pwd)
    return False

def ppwd():
    pwd=os.environ["PWD"]
    print(f"pwd = {pwd}")

def get_ds_prop(dataset, prop, default=None):
    if not dataset in spec['datasets']:
        return default
    if not prop in spec['datasets'][dataset]:
        return default
    return spec['datasets'][dataset][prop]
    
    
def get_dataset_dataframe(dataset, reclean=False):
    if not get_dataset(dataset, reclean=reclean):
        return False
    format =   spec['datasets'][dataset]['format']
    datafile_fullpath = root + "/datasets/" + dataset + "/" + spec['datasets'][dataset]['datafile']

    print(f"Load datafile {datafile_fullpath}")
    if format == 'csv':
        return pd.read_csv(datafile_fullpath)
    elif format == "xls":
        return pd.read_excel(datafile_fullpath,
                             sheet_name=get_ds_prop(dataset, 'sheet'),
                             indexcol=get_ds_prop(dataset, 'indexcol'),
                             skiprows=get_ds_prop(dataset, 'skiprows'))
    else:
        print(f"Abort: [{format}] format not supported for dataset {dataset}")
        return False

def print_dataset_info(dataset, reclean):
    df = get_dataset_dataframe(dataset, reclean=reclean)
    if not isinstance(df, pd.DataFrame):
        print(f"Abort: Could not load dataset [{dataset}]")
        exit(1)
    print(df.head())
    print(f"Shape: {df.shape}")              
    print(df.describe())
    target_attrib = get_ds_prop(dataset, 'target_attrib')
    if target_attrib:
        print( f"Target Attrib [{target_attrib}] info:")
        print(df[target_attrib].value_counts())


def get_pkl_path(dataset, name):
    root_dir = get_dataset_dir(dataset)
    return f"{root_dir}/{dataset}/{name}.pkl"
    
def serialize_dataframe(dataset, df, name):
    pkl_path = get_pkl_path(dataset, name, compress=1)
    joblib.dump(df, pkl_path)
    return True

def serialize_dataframe(dataset, df, name):
    pkl_path = get_pkl_path(dataset, name)
    df = joblib.load(pkl_path)
    return df
    
def discretize_attribute(df, attribute, buckets=6):
    discretized_attribute = attribute + "_discrete"
    #fmean=df[attribute].mean()
    fmax=df[attribute].max()
    fmin=df[attribute].min()
    rv = df[discretized_attribute] = np.ceil((buckets-1) * (df[attribute] - fmin) / (fmax - fmin))
    return rv


        
def discretize_attributes_per_spec(dataset, df, buckets=6):
    attributes = get_ds_prop(dataset, 'discritize_attributes')
    if attributes:
        for attribute in attributes:
            rv = discretize_attribute(df, attribute, buckets)
        return rv
    else:
        return df



        
def get_raw_training_and_test_data(dataset, df=None, rerun=False, reclean=False):
    """return X_train, y_train, X_test, y_test"""


    # load data if already saved
    dataset_path = root + "/datasets/" + dataset 
    X_train_file = dataset_path + "/X_train_file.pkl"
    y_train_file = dataset_path + "/y_train_file.pkl"
    X_test_file = dataset_path + "/X_test_file.pkl"
    y_test_file = dataset_path + "/y_test_file.pkl"
    all_files = [X_train_file, X_test_file, y_train_file, y_test_file]
    if all([os.path.exists(x) for x in all_files]) and not rerun:
           X_train = joblib.load(X_train_file)
           X_test  = joblib.load(X_test_file)
           y_train = joblib.load(y_train_file)
           y_test  = joblib.load(y_test_file)
           return X_train, y_train, X_test, y_test

    target_attribute = get_ds_prop(dataset, 'target_attrib')
    num_attribs = get_ds_prop(dataset, 'numerical_attributes')
    cat_attribs = get_ds_prop(dataset, 'categorical_attributes')

    
    
    if not target_attribute:
                raise(ValueError(f"Error: Dataset [{dataset}] is missing target_attrib property in spec.yml"))
    if not num_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing numerical_attributes property in spec.yml"))
    if not cat_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing categorical_attributes property in spec.yml"))

    if not df:
        df = get_dataset_dataframe(dataset, reclean=reclean)
        #print(f"Dataset {dataset} shape before split {df.shape}")
    synth_ops = get_ds_prop(dataset, 'synthesize_attributes', [])
    source_attribs = []
    for op in synth_ops:
        method = op.get('method')
        if not method:
            raise( ValueError( f"in dataset {dataset}: synthesize_attributes with no method specified."))
        source_attribs.append( op.get('source_attrib') )


    keep_attributes = num_attribs + cat_attribs + [target_attribute] + source_attribs
    drop_attributes = list( set(list(df.columns.values)) - set(keep_attributes) )
    for attribute in drop_attributes:
        print(f"Dropping attrib [{attribute}]")
        del df[attribute]


    #col_list=list(num_attribs + cat_attribs)
    #col_list.append(target_attribute)

    #print(f"BB> col_list = {col_list}")

    # tps = dict()
    # for x in df['outcome_type']:
    #     #print(x)
    #     tp = str(type(x))
    #     #print(tp)
    #     if tp in tps:
    #         tps[tp] += 1
    #     else:
    #         tps[tp] = 1
    #     print(tps)
    # df['outcome_type'].value_counts()

    # drop rows with missing data
    df=df.dropna() #(subset=col_list)     #drop all rows that have any NaN values, treat these training examples as suspect

    # add index column
    df=df.reset_index()

    #print( df[target_attribute].value_counts() )

    ## drop unwanted attributes
    #attributes = get_ds_prop(dataset, 'drop_attributes')
    #if attributes:
    #       for attribute in attributes:
    #           del df[attribute]
    


    ## synthesize attributes
    #synth_ops = get_ds_prop(dataset, 'synthesize_attributes', [])
    for op in synth_ops:
        method = op.get('method')
        if not method:
            raise( ValueError( f"in dataset {dataset}: synthesize_attributes with no method specified."))
        source_attrib = op.get('source_attrib')
        if not source_attrib:
            raise( ValueError( f"in dataset {dataset}: synthesize_attributes with no source_attrib specified."))
        new_attrib = op.get('new_attrib')
        if not new_attrib:
            raise( ValueError( f"in dataset {dataset}: synthesize_attributes with no new_attrib specified."))


        if method == 'regex_match_threshold':
            new_attrib = op.get('new_attrib')
            if not source_attrib:
                raise( ValueError( f"in dataset {dataset}: synthesize_attributes with no new_attrib specified."))
            regex = op.get('regex')
            if not regex:
                raise( ValueError( f"in dataset {dataset}: synthesize_attributes for source attrib {source_attrib} and method {method} -- no regex specified." ))
            newval_positive = op.get('newval_if_match','+')
            newval_negative = op.get('newval_if_nomatch','-')
            df[new_attrib] = df[source_attrib].map(lambda oldval: newval_positive if re.search(regex, oldval) else newval_negative)


        else:
            raise( ValueError( f"in dataset {dataset}: synthesize_attributes with unimplemented method [{method}] specified; cannot proceed." ))

    ## split data to training and test data, use stratified sampling to avoid sampling bias


    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df[target_attribute] = le.fit_transform(df[target_attribute].astype(str))


    from sklearn.model_selection import StratifiedShuffleSplit
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df[target_attribute]):
           strat_train_set = df.loc[train_index]
           strat_test_set = df.loc[test_index]

    X_train = strat_train_set.drop(target_attribute, axis=1) # drop labels for training set
    y_train = strat_train_set[target_attribute].copy()
    X_test  = strat_test_set.drop(target_attribute, axis=1) # drop labels for test set
    y_test  = strat_test_set[target_attribute].copy()

    # dump data to disk
    joblib.dump(X_train, X_train_file)
    joblib.dump(X_test,  X_test_file)
    joblib.dump(y_train, y_train_file)
    joblib.dump(y_test,  y_test_file)

    # return data
    return X_train, y_train, X_test, y_test

def get_attribs(dataset):
    num_attribs = get_ds_prop(dataset, 'numerical_attributes')
    cat_attribs = get_ds_prop(dataset, 'categorical_attributes')

    if not num_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing numerical_attributes property in spec.yml"))


    if not cat_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing categorical_attributes property in spec.yml"))

    
def get_dataprep_pipeline(dataset):
    num_attribs = get_ds_prop(dataset, 'numerical_attributes')
    cat_attribs = get_ds_prop(dataset, 'categorical_attributes')

    if not num_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing numerical_attributes property in spec.yml"))


    if not cat_attribs:
        raise(ValueError(f"Error: Dataset [{dataset}] is missing categorical_attributes property in spec.yml"))

        

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import Imputer
    
    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('std_scaler', StandardScaler()),
        ])
    
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
        ])

    from sklearn.pipeline import FeatureUnion
    dataprep_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

    return dataprep_pipeline





def eval_X_impedence_mismatch(dataset, X_train, X_train_prepared, X_test, X_test_prepared):
    print(type(X_train))
    print(type(X_train_prepared))
    cats = get_ds_prop(dataset, 'categorical_attributes')
    print(f"cats: {cats}")
    for cat in cats:
        train_has = set(X_train[cat])
        test_has =  set(X_test[cat])
        common_cats = train_has & test_has
        train_extra = train_has - common_cats
        test_extra = test_has - common_cats
        if len(train_extra) != 0:
            print(f"IN dataset {dataset}, X_train[{cat}] has extra members {train_extra}")
        if len(test_extra) != 0:
            print(f"IN dataset {dataset}, X_test[{cat}] has extra members {test_extra}")

        if len(train_extra) == 0 and len(test_extra) == 0:
            print(f"... ok |{dataset}.{cat}| == {len(common_cats)}  {train_extra}-{test_extra}")

        #print(f"common {dataset}.{cat}={common_cats}")
            
    raise(ValueError(f"!! Dataset {dataset} : X_train_prepared cols ({X_train_prepared.shape[1]}) != X_test_prepared cols ({X_test_prepared.shape[1]})"))
    pass




def get_prepared_training_and_test_data(dataset, df=None, rerun=False, reclean=False, x_transformer=None):
    from sklearn import preprocessing
    X_train, y_train, X_test, y_test = get_raw_training_and_test_data(dataset, df=None, rerun=rerun, reclean=reclean)
    if X_train.shape[1] != X_test.shape[1]:
        raise(ValueError(f"Dataset {dataset} : X_train cols ({X_train.shape[1]}) != X_test cols ({X_test.shape[1]})"))
    #else:
        #print(f"Carry on... {dataset} : X_train cols ({X_train.shape[1]}) == X_test cols ({X_test.shape[1]})")
    
    dataprep_pipeline1 = get_dataprep_pipeline(dataset) 
    X_train_prepared = dataprep_pipeline1.fit_transform(X_train)
    dataprep_pipeline2 = get_dataprep_pipeline(dataset) 
    X_test_prepared = dataprep_pipeline2.fit_transform(X_test)

    if X_train_prepared.shape[1] != X_test_prepared.shape[1]:
        eval_X_impedence_mismatch(dataset, X_train, X_train_prepared, X_test, X_test_prepared)
        raise(ValueError(f"Dataset {dataset} : X_train_prepared cols ({X_train_prepared.shape[1]}) != X_test_prepared cols ({X_test_prepared.shape[1]})"))
    #else:
    #    print(f"Carry on... {dataset} : X_train_prepared cols ({X_train_prepared.shape[1]}) == X_test_prepared cols ({X_test_prepared.shape[1]})")


    # balance it out...
    samplers = get_ds_prop(dataset, 'samplers')
    if samplers:
        for sampler_code in samplers:
            print("resampling X_test with "+sampler_code)
            sampler = eval(sampler_code)
            print(sampler)
            X_test_prepared, y_test = sampler.fit_sample(X_test_prepared, y_test)

        for sampler_code in samplers:
            print("resampling X_train with "+sampler_code)
            sampler = eval(sampler_code)
            X_train_prepared, y_train = sampler.fit_sample(X_train_prepared, y_train)

    # standardize (0 mean, 1 variance)
    X_train_scaled = preprocessing.scale(X_train_prepared)
    X_test_scaled  = preprocessing.scale(X_test_prepared)

    if x_transformer:
        X_train_final = x_transformer(X_train_scaled)
        X_test_final  = x_transformer(X_test_scaled)
    else:
        X_train_final = X_train_scaled
        X_test_final = X_test_scaled
    #print("HELLO")
    return X_train_final, y_train, X_test_final, y_test

     
# def get_classifiers_assignment1():
#     #You should implement five learning algorithms. They are for:

#     #Decision trees with some form of pruning
#     from sklearn import tree

#     #Neural networks - http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html
#     from sklearn.neural_network import MLPClassifier
    
#     #Boosting
    
    
#     #Support Vector Machines - http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
#     from sklearn.svm import SVC
    
#     #k-nearest neighbors - http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#     from sklearn.neighbors import KNeighborsClassifier
#     k=12
#     rv = [
#         [f"Multilayer Perceptron (neural network) []", MLPClassifier(hidden_layer_sizes=[100,100,100], learning_rate='adaptive', early_stopping=True)],
#         [f"Decision tree", tree.DecisionTreeClassifier()],
#         [f"kNearest Neighbors [k={k}, weights=uniform]",  KNeighborsClassifier(k, weights='uniform')],
#         [f"kNearest Neighbors [k={k}, weights=distance]", KNeighborsClassifier(k, weights='distance')]
#         ]
#     return rv

# def gen_learning_curves(dataset):
#     from sklearn.model_selection import learning_curve

#     X_train, y_train, X_test, y_test = get_training_and_test_data(dataset, df=None, rerun=False, reclean=False)
#     dataprep_pipeline = get_dataprep_pipeline(dataset)
#     X_train_prepared = dataprep_pipeline.fit_transform(X_train)
    
#     #train_sizes, train_scores, valid_scores = learning_curve(    clf, X_train_prepared, y_train, train_sizes=[.2,.4,.6,.8,1.0], cv=5)
#     outdir=get_results_dir(dataset)
#     # http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
#     for item in get_classifiers_assignment1():
#         classifier_name, clf = item
#         title=f"Learning Curve: {classifier_name}"
#         print(f"Below time is for [{title}]")
#         common_file = sanitize_filename(f"{outdir}/{dataset}_{title}")
#         png_file = common_file+".png"
#         pdf_file = common_file+".pdf"
#         timeit(lambda: plot_learning_curve(clf, title, X_train_prepared, y_train,ylim=(0.01, 1.01), save_png=png_file, save_pdf=pdf_file), prefix=title)





# validation curve - http://scikit-learn.org/stable/modules/learning_curve.html

def get_logfile(dataset, activity):
    return f"{root}/results/{dataset}.{activity}.log"


def optimize_hyperparameters(dataset, n_jobs=1, cv=4, algo=None, flat_results_dir=False, scoring=None, force=False):

    if not scoring:
        raise(ValueError("Unexpected - no scoring specified to optimize_hyperparameters"))
    
    X_train_prepared, y_train, X_test, y_test = get_prepared_training_and_test_data(dataset, df=None, rerun=False, reclean=False)

    if not scoring:
        raise(ValueError(f"dataset.{dataset}.scoring is not set, aborting hyperparameter optimization"))

    if flat_results_dir:
        outdir = get_results_dir(None)
    else:
        outdir = get_results_dir(dataset)
    

    n_iter = int(get_hyperparam_option(algo, 'n_iter'))
    clf_proc_name = spec['classifiers'][algo] 
    print(f"Now running search on {dataset} for {algo} with {clf_proc_name}")
    clf = eval( clf_proc_name )

    distribs = get_hyperparam_distrib(algo)
    print(f"  across hyperparameter distributions {distribs}")

    from sklearn.model_selection import RandomizedSearchCV
    searcher =  RandomizedSearchCV(clf, random_state=42, n_jobs=n_jobs, 
                                   param_distributions=distribs, cv=cv,
                                   n_iter=n_iter, scoring=scoring,
                                   verbose=True)
    print(f"searcher is {searcher}")

    searcher_file = f"{outdir}/{dataset}.{algo}.{scoring}.cvsearch.pkl"
    best_model_file = f"{outdir}/{dataset}.{algo}.{scoring}.best_model.pkl"
    if os.path.exists(searcher_file) and os.path.exists(best_model_file) and not force:
        print(f" files exist, skipping hyperparemeter optimization for {algo} (files are: {searcher_file} and {best_model_file}")
    else:
        timeit(lambda: searcher.fit(X_train_prepared, y_train), f"+++ hyperparam search of {dataset}.{algo}.{scoring}")
        joblib.dump(searcher, searcher_file)
        best_model = searcher.best_estimator_
        joblib.dump(best_model, best_model_file)
        cvres = searcher.cv_results_
        print("cvres:")
        print(cvres)



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

def get_test_evaluation(dataset, algo, scoring):

    model = get_bestfit_model(dataset, algo, scoring)
    target_attrib = get_ds_prop(dataset, 'target_attrib')

    X_train, y_train, X_test, y_test = get_prepared_training_and_test_data(dataset)

    print(f"X_train.shape={X_train.shape} X_test.shape={X_test.shape}")

    return evaluate_model_data(model, X_test, y_test)




