# https://python-packaging.readthedocs.io/en/latest/minimal.html
from setuptools import setup

setup(name='seobin_logger',
      version='0.1.0',
      description='Some customizable logger',
      url='http://github.com/parkseobin/seobin_logger',
      author='Seobin Park',
      author_email='seobinpark@hanyang.ac.kr',
      license='MIT',
      # Fill requirements!!
      install_requires=[
            'tensorboard', 
            'tensorboardX', 
            'tqdm>=4.48.2', 
            'numpy', 
            'openpyxl'
      ],
      packages=['seobin_logger'],
      zip_safe=False)