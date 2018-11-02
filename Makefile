SHELL=/bin/bash

# change this to match the name of python on $PATH which is version 3.6 or later.
python_exe=python3.6

datasets_dir=$(PWD)/datasets
pips=jupyter scipy pandas matplotlib scikit-learn kaggle seaborn numpy pyyaml google-cloud-bigquery xlrd graphviz geopy kaggle

venv_name=venv
with_venv=source $(venv_name)/bin/activate &&

datasets=ds1 ds4

#MENU menu: show this menu
menu:
	@cat $(PWD)/Makefile | grep '^#MENU ' | sed 's/^#MENU \+//'


tf/tf:
	mkdir -p tf
	touch $@

tf/python3-exists: tf/tf
	which $(python_exe)
	touch $@

tf/$(venv_name): tf/python3-exists
	python3.6 -m venv $(venv_name)
	$(with_venv) pip install --upgrade pip
	$(with_venv) pip install --upgrade $(pips)
	touch $@

#MENU notebook:   start jupyter notebook .. connect to http://thishost:8888 with password mlai
notebook: tf/$(venv_name)
	$(with_venv) cd $(PWD)/notebooks && JUPYTER_CONFIG_DIR=$(PWD) jupyter notebook --no-browser -y

#MENU getall:     get all datasets
getall_deps=$(addprefix tf/get-, $(datasets))
getall: $(getall_deps)
tf/get-%: tf/$(venv_name)
	$(with_venv)  bin/dataset_util.py -a get -s $*
	touch $@



#MENU tidy:       clean up ~ files
tidy:
	find -name \*~ -print0 | xargs -0 rm -f

#MENU pristine:   return workspace to initial state
pristine:
	rm -rf $(datasets_dir) $(venv_name) tf

clust:
	$(with_venv) bin/cluster_util.py -s ds1 -a kmeans-graph

clust2:
	$(with_venv) bin/cluster_util.py -s ds1 -a kmeans


kmeans-sweepk-CH-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a kmeans-sweepk-CH |& tee $@.log

kmeans-sweepk-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a kmeans |& tee $@.log

gmm-sweepk-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a gmm-sweepk |& tee $@.log

gmm-sweepk-CH-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a gmm-sweepk-CH |& tee $@.log



plot-kmeans-ksweep-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a kmeans-plot-ksweep

plot-gmm-ksweep-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a gmm-plot-ksweep

plot-gmm-ksweep-CH-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a gmm-plot-ksweep-CH

plot-kmeans-clusters-%:
	$(with_venv) time unbuffer bin/cluster_util.py -s $* -a kmeans-plot-clusters
