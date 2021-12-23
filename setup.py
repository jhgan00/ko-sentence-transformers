from glob import glob
from os.path import basename, splitext
from setuptools import find_packages, setup

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='ko_sentence_transformers',
    version='0.3',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    author="Junghyun Gan",
    author_email="jhgan00@yonsei.ac.kr",
    install_requires=['sentence-transformers'],
    license="Apache2",
    python_requires=">=3",
    long_description=long_description,
    long_description_content_type="text/markdown",
    py_modules=[splitext(basename(path))[0] for path in glob('src/ko_sentence_transformers/*.py')],
)