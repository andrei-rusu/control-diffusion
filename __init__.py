# Define the metadata attributes
__version__ = '0.1.0'
__author__ = 'Andrei C. Rusu (andrei-rusu)'
__license__ = 'MIT'
__description__ = 'A Python package containing agents that can rank nodes in a network for the purpose of controlling diffusion processes.'
__all__ = ['agent', 'general_utils', 'Agent']


from .control_diffusion import agent, general_utils, Agent