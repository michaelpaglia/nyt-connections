from setuptools import setup, find_packages

setup(
    name='nyt-connections-model',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'scipy',
        'gensim',
        'numpy',
    ],
)
