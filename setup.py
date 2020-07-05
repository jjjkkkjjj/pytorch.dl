from distutils.core import setup
import os


packages = ['dl']
for root,dirs,_ in os.walk('dl'):
    for d in dirs:
        if d not in ['__pycache__']:
            packages += [os.path.join(root,d).replace('/', '.')]

with open('envs/requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='pytorch.dl',
    version='0.0.1',
    packages=packages,
    install_requires=required,
    url='https://github.com/jjjkkkjjj/pytorch.dl',
    license='MIT',
    author='jjjkkkjjj',
    author_email='',
    description='Deep Learning Implementation with PyTorch.'
)
