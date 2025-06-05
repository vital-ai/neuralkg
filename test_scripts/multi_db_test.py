#!/usr/bin/env python
"""
Test script to demonstrate that multiple DatalogDatabase instances
can exist independently with different frame implementations.
"""
import os
import logging
import sys
from pathlib import Path
import pprint

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Import required modules
from neuralkg.datalog.engine.frame_factory import FrameFactory
from neuralkg.datalog.engine.database import DatalogDatabase
from neuralkg.datalog.engine.frame_types import FrameImplementation

def main():
    """
    Create multiple database instances with different implementations
    and show they can coexist independently.
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)s] %(message)s')
    logger = logging.getLogger(__name__)
    
    # Step 1: Load config file
    config_file = os.path.join(project_root, "config", "datalog.yaml")
    if os.path.exists(config_file):
        logger.info(f"Loading configuration from {config_file}")
        FrameFactory.load_config_from_file(config_file)
    else:
        logger.warning(f"Config file {config_file} not found, using defaults")
        
    # Step 2: Create database instances with specified implementations
    logger.info(f"Creating {FrameImplementation.PANDAS.value} database...")
    db1 = FrameFactory.make_database(FrameImplementation.PANDAS)
    
    # Try to create a second database with a different implementation
    # Note: This will fail with an exception if the implementation isn't available
    second_implementation = FrameImplementation.CUDF
    logger.info(f"Creating a second database with {second_implementation.value} implementation...")
    try:
        db2 = FrameFactory.make_database(second_implementation)
        second_impl_available = True
    except (ImportError, ValueError) as e:
        logger.warning(f"Could not create {second_implementation.value} database: {e}")
        logger.info(f"Creating another {FrameImplementation.PANDAS.value} database instead for demonstration purposes...")
        db2 = FrameFactory.make_database(FrameImplementation.PANDAS)
        second_impl_available = False
    
    # Step 3: Add some data to each database
    logger.info("Adding data to first database (pandas)...")
    add_test_data_to_db(db1, "db1")
    
    logger.info("Adding data to second database...")
    add_test_data_to_db(db2, "db2")
    
    # Step 4: Show that the databases are independent
    logger.info("\n=== Database 1 (pandas) ===")
    print(f"Implementation: {db1._implementation_name}")
    print_db_contents(db1)
    
    logger.info("\n=== Database 2 ===")
    print(f"Implementation: {db2._implementation_name}")
    print_db_contents(db2)
    
    # Step 5: Change global implementation and confirm databases are unaffected
    logger.info("\nChanging global FrameFactory implementation to 'pandas'...")
    FrameFactory.set_implementation("pandas")
    
    logger.info("\n=== After global implementation change ===")
    logger.info("Database 1:")
    print(f"Implementation: {db1._implementation_name}")
    print_db_contents(db1)
    
    logger.info("\nDatabase 2:")
    print(f"Implementation: {db2._implementation_name}")
    print_db_contents(db2)
    
    logger.info("\nTest complete!")

def add_test_data_to_db(db, prefix):
    """Add some test data to the given database."""
    # Create a relation
    db.create_relation(f"{prefix}_persons", 2, ("name", "age"))
    
    # Add facts using Datalog syntax
    db.load_program_from_string(f"""
    {prefix}_persons("Alice", 30).
    {prefix}_persons("Bob", 25).
    {prefix}_persons("Charlie", 35).
    """)

def print_db_contents(db):
    """Print the contents of all relations in the database."""
    for pred in db._relations:
        frame = db.get_relation(pred)
        print(f"\nRelation: {pred}")
        print(f"Columns: {frame.columns()}")
        print(f"Rows: {frame.num_rows()}")
        rows = frame.to_records()
        for row in rows:
            print(f"  {row}")

if __name__ == "__main__":
    main()
