import sys
import os
import pprint
import logging
from neuralkg.datalog.parser.datalog_parser import DatalogParser
from neuralkg.datalog.engine.database import DatalogDatabase
from neuralkg.datalog.engine.evaluator import BottomUpEvaluator


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
    print("[datalog_eval] Starting Datalog evaluation demo.")
    db = DatalogDatabase()

    # Load a datalog file into the database
    datalog_file = "test_data/dl1_test.datalog"
    print(f"Loading Datalog program from {datalog_file}")
    db.load_program_from_file(datalog_file)
    print("Loaded all facts and rules into DatalogDatabase.")

    evaluator = BottomUpEvaluator(db)
    evaluator.evaluate()
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
