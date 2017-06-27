#!/usr/bin/env python

from setuptools import setup

VERSION="0.1"

setup(
        name="label-extraction",
        version=VERSION,
        description="Wren Lab GEO label extraction and validation tools",
        author="Cory Giles, Chris Reighard, Xiavan Roopnarinesingh, Aleksandra Perz, Chase Brown, Hunter Porter",
        author_email="xiavan_roopnarinesing@ouhsc.edu",
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
        scripts=["bin/ale"],
        dependency_links=["https://gitlab.com/wrenlab/wrenlab.git"]
        )

