# Define the metadata attributes
__version__ = '0.1.0'
__author__ = 'Andrei C. Rusu (andrei-rusu)'
__license__ = 'MIT'
__description__ = \
        """
        A Python package containing agents that can be used to rank nodes in a network for the purpose of 
        limiting diffusion processes.
        """

# import exposed submodules and classes
from .src.agent import Agent
from .src import general_utils

import importlib.util as _importlib_util

def _lazy_import(name):
    # create a lazy loader for the submodule
    fullname = f'lib.agent.src.{name}' # use the absolute module name
    loader = _importlib_util.find_spec(fullname).loader
    lazy_loader = _importlib_util.LazyLoader(loader)
    # return a proxy object that will import the submodule on demand
    return lazy_loader.load_module(fullname) # pass the absolute module name

# expose the submodules from the src subpackage as attributes of the agent package
agent = _lazy_import('agent')
rank_model = _lazy_import('rank_model')

del _importlib_util, _lazy_import

__all__ = ['Agent', 'general_utils', 'agent', 'rank_model']