#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='demo2program',
      description='Neural Program Synthesis from Diverse Demonstration Videos',
      url='https://github.com/shaohua0116/demo2program',
      packages=find_packages(),
      scripts=[
      ],
      install_requires=[
          'tensorflow==1.15.5',
          'protobuf==3.20.2',
          'scipy',
          'numpy',
          'colorlog',
          'h5py',
          'Pillow',
          'progressbar',
          'ply',
          'tqdm',
          'joblib',
          'matplotlib'
      ],
      extras_require={

      },
      p_safe=True)
