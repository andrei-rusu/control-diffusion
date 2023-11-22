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
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    packages=find_packages(),
    python_requires='>=3.9, <3.10',
    install_requires=[
        'matplotlib', 
        'numpy', 
        'networkx',
    ],
    extras_require={
        'learn': [
            'pandas',
            'scikit-learn',
            'torch',
            'torch_scatter',
            'torch_sparse',
            'torch_geometric',
        ],
    },
)