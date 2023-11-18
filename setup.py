from setuptools import setup, find_packages

setup(
    name='control-diffusion',
    version='0.1.0',
    author='Andrei C. Rusu (andrei-rusu)',
    license='MIT',
    description=
        """
        A Python package containing agents that can be used to rank nodes in a network for the purpose of 
        limiting diffusion processes.
        """,
    packages=find_packages(),
    install_requires=[
        'matplotlib', 
        'numpy', 
        'scikit-learn', 
        'networkx'
    ],
)