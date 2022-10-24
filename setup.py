"""
DISTRIBUTION STATEMENT A. Approved for public release. Distribution is unlimited.

This material is based upon work supported by the Under Secretary of Defense for
Research and Engineering under Air Force Contract No. FA8702-15-D-0001. Any opinions,
findings, conclusions or recommendations expressed in this material are those of the
author(s) and do not necessarily reflect the views of the Under Secretary of Defense
for Research and Engineering.

© 2022 Massachusetts Institute of Technology.

The software/firmware is provided to you on an As-Is basis

Delivered to the U.S. Government with Unlimited Rights, as defined in DFARS Part
252.227-7013 or 7014 (Feb 2014). Notwithstanding any copyright notice, U.S. Government
rights in this work are defined by DFARS 252.227-7013 or DFARS 252.227-7014 as detailed
above. Use of this work other than as specifically authorized by the U.S. Government
may violate any copyrights that exist in this work.
"""

from setuptools import setup, find_packages

DISTNAME     = "pyLIQTR"
LICENSE      = 'BDS-2'
AUTHOR       = 'Joe Belarge, Arthur Kurlej, Justin Elenewski, John Blue'
AUTHOR_EMAIL = 'Joseph.Belarge@ll.mit.edu'
DESCRIPTION  = 'A python package for generating quantum circuits using quantum algorithms.'

setup(
    name=DISTNAME,
    version='0.2.0',
    license=LICENSE,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=[
        'numpy',
        'cirq',
        'pandas',
        'scipy',
        'matplotlib',
        'tqdm',
        'openfermion',
        'pyscf',
        'portalocker'
    ]
)