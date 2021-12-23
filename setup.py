from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

setup(
    name='ko_sentence_transformers',
    version='0.1',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author="Junghyun Gan",
    author_email="jhgan00@yonsei.ac.kr",
    install_requires=['sentence-transformers'],
    license="Apache2",
    python_requires=">=3",
    py_modules=[splitext(basename(path))[0] for path in glob('src/ko_sentence_transformers/*.py')],
)