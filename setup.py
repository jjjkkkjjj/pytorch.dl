from distutils.core import setup
import os

with open(os.path.join('envs', 'requirements.txt')) as f:
    install_requires = f.read().splitlines()

setup(
    name='pytorch.dl',
    version='0.0.1',
    packages=['dl', 'dl.core', 'dl.core.boxes', 'dl.train', 'dl.models', 'data', 'data.datasets',
              'data.augmentations'],
    install_requires=install_requires,
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
