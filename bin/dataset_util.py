#!/usr/bin/env python
import sys
from os.path import dirname,abspath
sys.path.append(dirname(__file__))
import pathlib

import yaml
from utils import *

import argparse

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description = 'dataset retreiver utility')
    parser.add_argument("-a", "--action", required=True, choices=["list","get","info"], help="action")
    parser.add_argument("-s", "--dataset", required=False, help="dataset name, eg: ds1)")
    parser.add_argument("-r", "--reclean", action="store_const", const=True, default=False, help="rerun data clean commands")    
    opts = parser.parse_args()

    if opts.action=="list":
        for ds in spec['datasets']:
            print(f"{ds}:    {spec['datasets'][ds]['name']}")
        exit(0)
    elif opts.action=="get":
        if not opts.dataset:
            sys.exit("Error: specify --dataset")
        if not opts.dataset in spec['datasets']:
            sys.exit(f"Error: detaset {opts.dataset} not defined in spec.yml")
        get_dataset(opts.dataset, reclean=opts.reclean)
    elif opts.action=="info":
        print_dataset_info(opts.dataset, reclean=opts.reclean)
    else:
        print(f"Action not implemented: {opts.action}")
        exit(1)
        
              
