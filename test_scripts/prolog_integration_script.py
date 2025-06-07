"""
Integration test for NeuralKG's Prolog integration via Janus.
"""
import sys
import os
import logging
import pandas as pd

# Add parent directory to system path so we can import the neuralkg package
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path if needed
if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import NeuralKG modules
from neuralkg.prolog import DatalogBridge, initialize_prolog

def test_basic_integration():
    """Test basic integration between Python and Prolog."""
    logger.info("Starting basic integration test...")
    
    try:
        # Initialize Prolog
        janus = initialize_prolog()
        logger.info("✅ Successfully initialized Prolog")
        
        # Test calling from Python to Prolog directly
        # Our module is already loaded during initialization_prolog
        result = janus.query_once("test_prolog_python('Hello from Python', Result)")
        logger.info(f"✅ Python -> Prolog -> Python call result: {result}")
        
        # Get the bridge instance
        from neuralkg.prolog.bridge import get_bridge
        bridge = get_bridge()
        
        # Create sample pandas dataframes
        person_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie', 'David'],
            'age': [25, 32, 45, 19],
            'city': ['New York', 'Boston', 'Chicago', 'Seattle']
        })
        
        product_df = pd.DataFrame({
            'product_id': [101, 102, 103, 104],
            'name': ['Laptop', 'Phone', 'Tablet', 'Headphones'],
            'price': [999.99, 699.99, 349.99, 149.99],
            'in_stock': [True, True, False, True]
        })
        
        # Register the dataframes with the bridge
        bridge.register_frame('person', person_df)
        bridge.register_frame('product', product_df)
        logger.info("✅ Registered sample dataframes")
        
        # Test 1: Direct Prolog call with static results
        query = "query_relation(person, ['Alice', 32], Results)"
        result = janus.query_once(query)
        logger.info(f"✅ Simple query result: {result}")
        
        # Test 2: Bidirectional call - Python -> Prolog -> Python -> Prolog
        # Prolog calls back to Python to query the dataframe and returns the result
        logger.info("Testing bidirectional Python-Prolog-Python dataframe queries:")
        
        # Query for Alice
        query = "query_pandas(person, ['Alice'], Results)"
        result = janus.query_once(query)
        logger.info(f"✅ Bidirectional person query (Alice): {result}")
        
        # Query for product data
        query = "query_pandas(product, [103], Results)"
        result = janus.query_once(query)
        logger.info(f"✅ Bidirectional product query (ID 103): {result}")
        
        # Query with a result that should return multiple rows
        query = "query_pandas(product, [], Results)"
        result = janus.query_once(query)
        logger.info(f"✅ Bidirectional product query (all products): {result}")
        
        # Test well-founded semantics
        query = "test_wfs(X)"
        result = list(janus.query(query))
        logger.info(f"✅ Well-founded semantics test result: {result}")
        
        return True
    
    except Exception as e:
        logger.error(f"❌ Test failed: {str(e)}", exc_info=True)
        return False



if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("NEURALKG PROLOG INTEGRATION TEST")
    logger.info("=" * 60)
    
    # Run integration test
    result = test_basic_integration()
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    logger.info(f"Integration test: {'PASSED' if result else 'FAILED'}")
    
    # Exit with appropriate code
    sys.exit(0 if result else 1)
