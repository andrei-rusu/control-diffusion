#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='control_diffusion',
    version='0.1.0',
    author='Andrei C. Rusu (andrei-rusu)',
    description='A Python package containing agents that can rank nodes in a network for the purpose of controlling diffusion processes.',
    long_description=open('README.md','rt').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/andrei-rusu/control-diffusion',
    project_urls={
        'Bug Tracker': 'https://github.com/andrei-rusu/control-diffusion/issues',
    },
    license='MIT',
    classifiers=[...],
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'matplotlib', 
        'numpy', 
        'scikit-learn', 
        'networkx'
    ],
    extras_require={
        'learn': [
            'torch',
            'torch_geometric',
        ],
    },
)