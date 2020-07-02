from distutils.core import setup

setup(
    name='pytorch.dl',
    version='0.0.1',
    packages=['dl', 'dl.core', 'dl.core.boxes',
              'dl.train',
              'dl.models', 'dl.models.ssd', 'dl.models.vgg',
              'dl.data', 'dl.data.datasets', 'dl.data.augmentations'],
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
