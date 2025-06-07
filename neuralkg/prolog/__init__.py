"""
SWI-Prolog integration module for NeuralKG.
"""
from .bridge import DatalogBridge, initialize_prolog, register_python_callbacks

__all__ = ['DatalogBridge', 'initialize_prolog', 'register_python_callbacks']
