from distutils.core import setup

import os
packages = ['dl']
for root,dirs,_ in os.walk('dl'):
    for d in dirs:
        if d not in ['__pycache__']:
            packages += [os.path.join(root,d).replace('/', '.')]


setup(
    name='pytorch.dl',
    version='0.0.1',
    # too stupid code....
    packages=packages,
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
