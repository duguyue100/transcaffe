"""Setup script for the transcaffe package.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3.4
Programming Language :: Python :: 3.5
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Scientific/Engineering :: Artificial Intelligence
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

try:
    from simretina import __about__
    about = __about__.__dict__
except ImportError:
    about = dict()
    exec(open("transcaffe/__about__.py").read(), about)

setup(
    name='transcaffe',
    version=about["__version__"],

    author=about["__author__"],
    author_email=about["__author_email__"],

    url=about["__url__"],

    packages=["transcaffe"],

    classifiers=list(filter(None, classifiers.split('\n'))),
    description="Transfer Caffe Model to HDF5 format."
)
