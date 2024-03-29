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

import os
from setuptools import setup, find_packages
from pathlib import Path

def read(rel_path: str) -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    # intentionally *not* adding an encoding option to open, See:
    #   https://github.com/pypa/virtualenv/issues/201#issuecomment-3145690
    with open(os.path.join(here, rel_path)) as fp:
        return fp.read()

def get_version(rel_path: str) -> str:
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            # __version__ = "0.9"
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError("Unable to find version string.")

DISTNAME = "pyLIQTR"
LICENSE = "BDS-2"
AUTHOR = "Kevin Obenland, Justin Elenewski, Arthur Kurlej,  Joe Belarge, John Blue, and Robert Rood"
AUTHOR_EMAIL = "Kevin.Obenland@ll.mit.edu"
DESCRIPTION = (
    "A python package for generating quantum circuits using quantum algorithms."
)
THIS_DIRECTORY = Path(__file__).parent
LONG_DESCRIPTION = (THIS_DIRECTORY / "README.md").read_text()

REQUIREMENTS = open(THIS_DIRECTORY / "requirements.txt").readlines()
REQUIREMENTS = [r.strip() for r in REQUIREMENTS]

REQUIREMENTS_DEV = open(THIS_DIRECTORY / "requirements-dev.txt").readlines()
REQUIREMENTS_DEV = [r.strip() for r in REQUIREMENTS_DEV]

__version__ = ""
exec(open("src/pyLIQTR/_version.py").read())

setup(
    name=DISTNAME,
    version=__version__,
    license=LICENSE,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8,<=3.12",
    install_requires=REQUIREMENTS,
    extras_require={"dev": REQUIREMENTS_DEV},
)
