#!/usr/bin/env python

from setuptools import setup

VERSION="0.1"

setup(
        name="aletools",
        version=VERSION,
        description="Wren Lab GEO automated label extraction and validation tools",
        author="Cory Giles, Chris Reighard, Xiavan Roopnarinesingh, Aleksandra Perz, Chase Brown, Hunter Porter",
        author_email="xiavan_roopnarinesingh@ouhsc.edu",
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
            "Natural Language :: English",
            "Operating System :: POSIX",
            "Programming Language :: Python :: 3.6",
            "Topic :: Scientific/Engineering :: Bio-Informatics"
            ],
        license="AGPLv3+",
        #packages=find_packages(),
        packages=["mle"],
        package_dir={"mle": "mle"},
        package_data={"mle": ['data/*.tsv']},
        include_package_data=True,
        scripts=["bin/ale","bin/ale-validation"],
        dependency_links=["https://gitlab.com/wrenlab/wrenlab.git"]
        )

