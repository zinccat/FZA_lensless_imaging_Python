# -*- coding: utf-8 -*-
"""
pycompsense: A toolbox for compressed sensing and sparse reconstruction algorithms.

Copyright (C) 2012 Amit Aides
Author: Amit Aides <amitibo@tx.technion.ac.il>
URL: <http://bitbucket.org/amitibo/pycompsense>
License: See attached license file
"""

from setuptools import setup

NAME = 'pycompsense'
PACKAGE_NAME = 'compsense'
PACKAGES = [PACKAGE_NAME, '%s.algorithms' % PACKAGE_NAME]
VERSION = '0.1.3'
DESCRIPTION = 'A toolbox for compressed sensing and sparse reconstruction algorithms'
LONG_DESCRIPTION = """
`pycompsense` is a toolbox for compressed sensing and sparse reconstruction algorithms.
It is based on `sparco <http://www.cs.ubc.ca/labs/scl/sparco/>`_.

`pycompsense` includes an implementation of `TwIST <http://www.lx.it.pt/~bioucas/TwIST/TwIST.htm>`_.
"""
AUTHOR = 'Amit Aides'
EMAIL = 'amitibo@tx.technion.ac.il'
KEYWORDS = ["singal processing", "sparse representation", "compressed sensing", "CS", 'optimization']
LICENSE = 'GPLv3'
CLASSIFIERS = [
    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    'Development Status :: 2 - Pre-Alpha',
    'Topic :: Scientific/Engineering'
]
URL = "http://bitbucket.org/amitibo/pycompsense"

def main():
    """main setup function"""
    
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        author=AUTHOR,
        author_email=EMAIL,
        url=URL,
        keywords=KEYWORDS,
        classifiers=CLASSIFIERS,
        license=LICENSE,
        packages=PACKAGES,
        package_data={PACKAGE_NAME: ['data/*.*']},
        install_requires=['pyrwt >= 0.2'],
        )


if __name__ == '__main__':
    main()
