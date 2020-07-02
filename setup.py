from distutils.core import setup

setup(
    name='pytorch.dl',
    version='0.0.1',
    # too stupid code....
    packages=['dl',
              'dl.train',
              'dl.models',
              'dl.models.ssd', 'dl.models.ssd.core', 'dl.models.ssd.core.boxes',
              'dl.models.vgg',
              'dl.data', 'dl.data.object', 'dl.data.object.datasets', 'dl.data.object.augmentations',
              'dl.data.text', 'dl.data.text.datasets'],
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
