## ALE CLI - Automated Label Extraction Command Line Interface

Tools to extract and validate labels using wrenlab's ALE.
Gold standard for validation is Wrenlab's JDW manual annotations. 

## REQUIREMENTS

- [wrenlab](https://gitlab.com/wrenlab/wrenlab)
- [metalearn](https://gitlab.com/wrenlab/metalearn)
- [click](https://pypi.python.org/pypi/click)

## INSTALL

python setup.py develop --user

## USAGE

The following scripts should be in your $PATH and accessible from your shell

```
ale
ale-validation
```

ale extracts labels from GEO. ale-validation evaluates performance against the Wrenlab gold standard for GEO data for human data (NCBI Taxon ID 9606). 
