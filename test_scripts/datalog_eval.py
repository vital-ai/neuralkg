import sys
import os
import pprint
import argparse
import logging
from neuralkg.datalog.parser.datalog_parser import DatalogParser
from neuralkg.datalog.engine.database import DatalogDatabase
from neuralkg.datalog.engine.frame_factory import FrameFactory
from neuralkg.datalog.engine.frame_types import FrameImplementation
from neuralkg.datalog.engine.config import config


def print_table(results, header=None):
    # Print query results using the Frame API
    if results is None:
        print("    [no results - query returned None]")
        return
        
    if not (hasattr(results, 'num_rows') and hasattr(results, 'columns') and hasattr(results, 'to_records')):
        print("    [ERROR: results object doesn't conform to Frame API]")
        logger.error(f"Print table received invalid results type: {type(results)}")
        return
        
    num_rows = results.num_rows()
    columns = results.columns()
    records = results.to_records()
    
    logger.debug(f"[PRINT_TABLE] num_rows: {num_rows}")
    logger.debug(f"[PRINT_TABLE] columns: {columns}")
    logger.debug(f"[PRINT_TABLE] records: {records}")
    
    if num_rows == 0:
        print("    [no results - empty frame]")
        return
        
    # Print header and rows for non-empty results
    print("    " + " | ".join(str(col) for col in columns))
    print("    " + "-+-".join('-' * len(str(col)) for col in columns))
    for row in records:
        print("    " + " | ".join(str(row.get(col, '')) for col in columns))

# Set up logger
logger = logging.getLogger("datalog")
log_level = os.environ.get("DLG_DEBUG", "WARNING").upper()
logger.setLevel(log_level)
ch = logging.StreamHandler()
ch.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] %(message)s')
ch.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(ch)

# Patch DatalogParser and DatalogTransformer to use logging (if not already done)
# If you want to globally suppress debug prints, set DLG_DEBUG=WARNING or ERROR

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Datalog Evaluation Script")
    parser.add_argument(
        "--config", 
        type=str,
        default="config/datalog.yaml",
        help="Path to configuration file")
    parser.add_argument(
        "--implementation",
        type=str,
        choices=[impl.value for impl in FrameImplementation],
        help=f"Override the Frame implementation to use. Available options: {FrameImplementation.get_all_implementations()}")
    parser.add_argument(
        "--datalog-file",
        type=str,
        default="test_data/dl1_test.datalog",
        help="Path to Datalog program file")
    
    args = parser.parse_args()
    
    print("[datalog_eval] Starting Datalog evaluation demo.")
    
    # Load configuration from file if specified
    if os.path.exists(args.config):
        print(f"Loading configuration from {args.config}")
        FrameFactory.load_config_from_file(args.config)
    else:
        print(f"Configuration file {args.config} not found. Using default configuration.")
    
    # Determine which implementation to use
    implementation_value = args.implementation  # May be None if not specified on command line
    
    if implementation_value:
        # Validate that it's a valid enum value
        try:
            # Convert string to enum value
            if FrameImplementation.is_valid(implementation_value):
                implementation = FrameImplementation(implementation_value)
            else:
                valid_options = FrameImplementation.get_all_implementations()
                raise ValueError(f"Invalid implementation: {implementation_value}. Valid options: {valid_options}")
            print(f"Using specified Frame implementation: {implementation.value}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    else:
        # Otherwise use the one from configuration
        FrameFactory.initialize()
        impl_name = FrameFactory.get_current_implementation_name()
        try:
            # Validate the implementation from config
            if FrameImplementation.is_valid(impl_name):
                implementation = FrameImplementation(impl_name)
            else:
                # This should not happen if config validation is proper
                valid_options = FrameImplementation.get_all_implementations()
                raise ValueError(f"Invalid implementation in config: {impl_name}. Valid options: {valid_options}")
            print(f"Using Frame implementation from config: {implementation.value}")
        except ValueError as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    # Create the database with the specified implementation using the factory pattern
    try:
        db = FrameFactory.make_database(implementation)
    except (ImportError, ValueError) as e:
        print(f"Error creating database with {implementation.value} implementation: {e}")
        sys.exit(1)

    # Load the datalog file into the database
    datalog_file = args.datalog_file
    print(f"Loading Datalog program from {datalog_file}")
    db.load_program_from_file(datalog_file)
    print("Loaded all facts and rules into DatalogDatabase.")

    # Evaluate rules using the database's evaluate method
    db.evaluate()
    print("Evaluation complete.")

    # DIAG: Print all rows for key relations
    for pred in ["emp_name", "employee3", "edge_id", "edge_source", "edge_target", "edge_weight", "light_edge"]:
        try:
            frame = db.get_relation(pred)
            print(f"\n[DIAG] Relation: {pred}")
            print(f"Columns: {frame.columns()}")
            print(f"Rows: {frame.num_rows()}")
            sample = frame.to_records()[:5]
            for row in sample:
                print(f"  {row}")
        except Exception as e:
            print(f"[DIAG] Error for {pred}: {e}")

    # DEBUG: Print all predicates, their schemas, and row counts
    logger.debug("All predicates after evaluation:")
    for pred in db.all_predicates():
        try:
            frame = db.get_relation(pred)
            logger.debug(f"  {pred}: columns={frame.columns()}, rows={frame.num_rows()}")
        except Exception as e:
            print(f"  {pred}: error getting relation: {e}")

    # DEBUG: Print all loaded rules
    logger.debug("Loaded rules:")
    for rule in db.rules:
        logger.debug(f"  {rule}")

    # Print all rows for key predicates to diagnose negation/projection
    def print_all_rows(pred):
        try:
            frame = db.get_relation(pred)
            logger.debug(f"Contents of {pred} (columns={frame.columns()}):")
            rows = frame.to_records()
            if not rows:
                print("    [no results]")
            else:
                for row in rows:
                    logger.debug(f"   {row}")
        except Exception as e:
            print(f"  {pred}: error getting relation: {e}")


    query_examples = [
        # Existing queries
        "light_edge(X, Y).",
        "light_edge3(X, Y, W).",
        "path(X, Y).",
        "eligible(Name).",
        # Basic edge-case queries
        # 1. Repeated variables in head
        # "emp_name(Emp, Name), emp_salary(Emp, Salary), Emp = Emp.",
        # 2. Variable appears only in comparison
        "emp_salary(Emp, Salary), Salary > 90000.",
        
        # Aggregate queries - test different aggregation functions
        # 3. Sum: Group total sales amount by employee name
        "emp_id(Emp), emp_name(Emp, Name), sale(Emp, Product, Amount, Quantity, Date), agg_sum(Name, Amount, TotalAmount).",
        
        # 4. Count: Count number of sales by product
        "sale(Emp, Product, Amount, Quantity, Date), agg_count(Product, Emp, SalesCount).",
        
        # 5. Max: Find maximum sale amount by product
        "sale(Emp, Product, Amount, Quantity, Date), agg_max(Product, Amount, MaxAmount).",
        
        # 6. Min: Find minimum sale amount by product
        "sale(Emp, Product, Amount, Quantity, Date), agg_min(Product, Amount, MinAmount).",
        
        # 7. Avg: Calculate average sale amount by employee name
        "emp_id(Emp), emp_name(Emp, Name), sale(Emp, Product, Amount, Quantity, Date), agg_avg(Name, Amount, AvgAmount).",
        
        # 8. Count with multiple grouping variables: Count sales by employee and product
        "emp_id(Emp), emp_name(Emp, Name), sale(Emp, Product, Amount, Quantity, Date), agg_count(Name, Product, SalesCount).",
        
        # 9. Sum with multiple grouping variables: Sum hours by employee and project status
        "emp_id(Emp), emp_name(Emp, Name), assignment(Emp, PID, Hours), project(PID, PName, Budget, Status), agg_sum(Name, Status, Hours, TotalHours).",
        
        # 10. Complex query with filtering: Sum project budgets by status
        "project(PID, Name, Budget, Status), agg_sum(Status, Budget, TotalBudget).",
        
        # 11. Test negation - should exclude employees with salaries
        "emp_name(Emp, Name), not emp_salary(Emp, Salary).",
        # 12. Ground query that should return true
        "emp_name('e_bob', 'bob').",
        # 13. Ground query that should return false (no results)
        "emp_name('e_bob', 'alice').",
        # Other test query examples (commented out)
        # "emp_name(Emp, 'alice'), emp_salary(Emp, Salary)",
        # "emp_salary(Emp, Salary), emp_name(Emp, Name)",
        # "emp_name(Emp, Name), emp_name(Emp, Name)",
        # "emp_salary(Emp, Salary), Salary = Salary",
        # "Salary > 100000"
        ]

    # --- Example queries ---

    for query_str in query_examples:
        print(f"\n-- Query: {query_str}")
        results = db.query_from_string(query_str)
        logger.debug(f"[MAIN] About to print results for query: {query_str}")
        logger.debug(f"[MAIN] Results columns: {results.columns() if hasattr(results, 'columns') else None}")
        logger.debug(f"[MAIN] Results sample rows: {results.to_records()[:5] if hasattr(results, 'to_records') else None}")
        print_table(results)

if __name__ == "__main__":
    main()
