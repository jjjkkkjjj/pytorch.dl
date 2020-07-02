from distutils.core import setup
import os


setup(
    name='pytorch.dl',
    version='0.0.1',
    packages=['dl', 'dl.core', 'dl.core.boxes', 'dl.train', 'dl.models', 'data', 'data.datasets',
              'data.augmentations'],
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
