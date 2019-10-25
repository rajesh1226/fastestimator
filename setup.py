import os
import re

from setuptools import find_packages, setup

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy


extensions = [
    Extension(
        'fastestimator.util.compute_overlap',
        ['fastestimator/util/compute_overlap.pyx'],
        include_dirs=[numpy.get_include()]
    ),
]


def get_version():
    path = os.path.dirname(__file__)
    version_re = re.compile(r'''__version__ = ['"](.+)['"]''')
    with open(os.path.join(path, 'fastestimator', '__init__.py')) as f:
        init = f.read()
    return version_re.search(init).group(1)


setup(
    name="fastestimator",
    version=get_version(),
    description="Deep learning Application framework",
    packages=find_packages(),
    package_dir={'': '.'},
    long_description="FastEstimator is a high-level deep learning API. With the help of FastEstimator, you can easily \
                    build a high-performance deep learning model and run it anywhere.",
    author="FastEstimator Dev",
    url='https://github.com/fastestimator/fastestimator',
    license="Apache License 2.0",
    keywords="fastestimator tensorflow",
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3", ],
    ext_modules    = cythonize(extensions),

    # Declare minimal set for installation
    install_requires=[
        'numpy',
        'pyfiglet',
        'pandas',
        'pillow',
        'sklearn',
        'wget',
        'matplotlib',
        'seaborn>= 0.9.0',
        'scipy',
        'pytest',
        'pytest-cov',
        'tensorflow-probability',
        'umap-learn',
        'tqdm',
        'opencv-python',
        'papermill',
        'tf-explain',
        'slackclient',
        'nest_asyncio',
	'pycocotools'	
    ],
    setup_requires = ["cython", "numpy"],
    # Declare extra set for installation
    extras_require={},
    scripts=['bin/fastestimator'])
