"""
Bridge module for SWI-Prolog and Python integration.
"""
import os
import sys
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)

class DatalogBridge:
    """
    Bridge between Python dataframes and Prolog predicates.
    This class manages the communication between Python and Prolog,
    allowing dataframe data to be queried from Prolog.
    """
    
    def __init__(self):
        """Initialize the DatalogBridge."""
        self.frames = {}  # Maps predicate names to dataframes

    def register_frame(self, predicate_name, df):
        """
        Register a dataframe for a specific predicate.
        
        Args:
            predicate_name: Name of the predicate this dataframe represents
            df: The pandas dataframe
        """
        self.frames[predicate_name] = df
        logger.info(f"Registered dataframe for predicate '{predicate_name}'")
    
    def query_dataframe(self, predicate, args):
        """
        Query a dataframe for facts matching the given predicate and arguments.
        
        Args:
            predicate: The name of the predicate to query
            args: A list of arguments (can contain variables as None)
            
        Returns:
            A list of matching facts as tuples
        """
        if predicate not in self.frames:
            logger.warning(f"No dataframe registered for predicate '{predicate}'")
            return []
            
        df = self.frames[predicate]
        logger.debug(f"Querying dataframe for {predicate}({args})")
        
        # Extract concrete arguments (not variables)
        concrete_args = [(i, arg) for i, arg in enumerate(args) if arg is not None]
        
        # Filter dataframe based on concrete arguments
        filtered_df = df
        for i, arg in concrete_args:
            if i < len(df.columns):
                filtered_df = filtered_df[filtered_df[df.columns[i]] == arg]
        
        # Convert filtered results to a list of tuples
        results = filtered_df.values.tolist()
        logger.debug(f"Query returned {len(results)} results")
        return results
        
    def test_callback(self, arg1, arg2):
        """Test callback function for Prolog to call."""
        logger.info(f"Test callback called with args: {arg1}, {arg2}")
        return f"Python received: {arg1} and {arg2}"


# Global bridge instance to be used by Janus callbacks
_bridge = DatalogBridge()

# For Janus 1.5.2, we need to expose callable objects as module attributes
# These wrappers will be accessible via py_call(neuralkg.prolog.bridge:fn_name, Result)

# Functions that will be called from Prolog must be callable objects
# with a __call__ method to work with Janus 1.5.2

# Test callback function for Prolog to call
class TestCallback:
    def __call__(self, arg1, arg2):
        logger.info(f"TestCallback called with args: {arg1}, {arg2}")
        return f"Python received: {arg1} and {arg2}"

# Query dataframe function for Prolog to call
class QueryDataframe:
    def __call__(self, predicate, args):
        logger.info(f"QueryDataframe called with predicate: {predicate}, args: {args}")
        return _bridge.query_dataframe(predicate, args)

# Expose as module attributes for Janus to access
# These will be accessed via py_call('neuralkg.prolog.bridge', 'test_callback', Fn) in Prolog
test_callback = TestCallback()
query_dataframe = QueryDataframe()

def get_bridge():
    """Get the global bridge instance."""
    return _bridge

def initialize_prolog(prolog_dir=None):
    """
    Initialize SWI-Prolog using Janus.
    
    Args:
        prolog_dir: Directory containing Prolog files (default: project's prolog dir)
    
    Returns:
        The janus module for interacting with Prolog
    """
    try:
        import janus_swi as janus
        logger.info("Janus-SWI module imported successfully")
        
        # Set up Prolog directory
        if prolog_dir is None:
            # Use default location in project
            prolog_dir = str(Path(__file__).parent.parent.parent / 'prolog')
        
        # Make sure Prolog can find our files
        janus.query_once(f"assertz(user:file_search_path(neuralkg_prolog, '{prolog_dir}'))")
        
        # Explicitly load our Prolog module
        prolog_file = os.path.join(prolog_dir, 'neuralkg.pl')
        logger.info(f"Loading Prolog module from: {prolog_file}")
        
        # Ensure the file exists
        if not os.path.exists(prolog_file):
            raise FileNotFoundError(f"Prolog module file not found: {prolog_file}")
            
        # Load the module
        result = janus.query_once(f"consult('{prolog_file}')")
        if result is False:
            logger.error(f"Failed to load Prolog module: {prolog_file}")
            raise RuntimeError(f"Failed to load Prolog module: {prolog_file}")
        
        logger.info(f"Successfully loaded Prolog module: {prolog_file}")
        
        # Register Python callbacks
        register_python_callbacks(janus)
        
        return janus
        
    except ImportError as e:
        logger.error(f"Failed to import janus_swi: {str(e)}")
        logger.error("Please install janus-swi with: pip install janus-swi")
        raise ImportError("janus-swi not installed. Run: pip install janus-swi") from e

def register_python_callbacks(janus):
    """
    Register Python functions that can be called from Prolog.
    
    Args:
        janus: The janus module
    """
    # In newer versions of janus-swi, Python callbacks are automatically registered
    # when they are called from Prolog with py_call/1 or py_call/2
    # No explicit registration is needed
    
    logger.info("Python callbacks will be automatically registered when called from Prolog")
