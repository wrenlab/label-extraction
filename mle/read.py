#!/bin/env python

import pandas as pd
import os

import wrenlab
from wrenlab.ontology import fetch

import labelext

def test():
    print("Hello world")
 
def get_ma():
    this_dir,this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir,"data","geo_manual_labels_jdw.tsv")
    ma = pd.read_csv(DATA_PATH,sep="\t",index_col=0)
    ma.index = [int(i.strip("GSM")) for i in ma.index]
    ma.index.name = "SampleID"
    return(ma)

def tissues_dict():
    BTO = fetch('BTO')
    return {BTO._resolve_id(int(i)):BTO.name_map()[i]
            for i in BTO.name_map().keys()}

def get_tissue():
    this_dir,this_filename = os.path.split(__file__)
    DATA_PATH = os.path.join(this_dir,"data","tissue.logistic_regression.pkl")
    path = DATA_PATH
    ti = pd.read_pickle(path)
    return(ti)

if __name__ == "__main__":
    test()
    ma = get_ma()
    ti = get_tissue()
    y,y_hat = labelext.make_y(ti,ma,96)
    print(y.head())
    print(y_hat.head())
    print(y.shape)
    print(y_hat.shape)
