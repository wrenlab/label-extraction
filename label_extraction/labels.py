#!/usr/bin/env python

import wrenlab
from wrenlab.ncbi.geo import label
import click

#suppress FutureWarnings lol
import warnings
warnings.simplefilter(action="ignore",category=FutureWarning)

@click.command()
def get_labels(taxonID=9606):
    #take input taxon id and default to stdout printing
    #command line arguments are taxon id, delimiters, and columns? 
    output = label.get_labels(taxon_id=taxonID)
    print(",".join([ str(x) for x in output.columns]))
    for line in output.itertuples():
        print(",".join([ str(x) for x in line]))

def validate():
    #print validation code
    #get from cory
    pass

if __name__ == "__main__":
    get_labels()
