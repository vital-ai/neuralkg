import sys
import os
import pprint
import logging
from neuralkg.datalog.parser.datalog_parser import DatalogParser
from neuralkg.datalog.engine.database import DatalogDatabase
from neuralkg.datalog.engine.evaluator import BottomUpEvaluator


def print_table(results, header=None):
    # Print a Frame (e.g., PandasFrame) in a readable table format using its public API
    if hasattr(results, 'num_rows') and hasattr(results, 'columns') and hasattr(results, 'to_records'):
        num_rows = results.num_rows()
        columns = results.columns()
        records = results.to_records()
    else:
        # fallback for pandas.DataFrame
        df = results
        num_rows = df.shape[0]
        columns = list(df.columns)
        records = df.to_dict(orient="records")
    logger.debug(f"[PRINT_TABLE] num_rows: {num_rows}")
    logger.debug(f"[PRINT_TABLE] columns: {columns}")
    logger.debug(f"[PRINT_TABLE] records: {records}")
    if num_rows == 0:
        print("    [no results]")
        return
    # Always print header and rows for non-empty results
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
        "light_edge(X, Y)",
        "light_edge3(X, Y, W)",
        "path(X, Y)",
        "eligible(Name)",
        # Additional edge-case queries
        # 1. Repeated variables in head
        "emp_name(Emp, Name), emp_salary(Emp, Salary), Emp = Emp",
        # 2. Variable appears only in comparison
        "emp_salary(Emp, Salary), Salary > 90000",
        # 3. Variable appears only in negation
        "emp_name(Emp, Name), not emp_salary(Emp, Salary)",
        # 4. Mix of constants and variables
        # "emp_name(Emp, 'alice'), emp_salary(Emp, Salary)",
        # 5. Permuted variable order
        #"emp_salary(Emp, Salary), emp_name(Emp, Name)",
        # 6. All variables repeated
        # "emp_name(Emp, Name), emp_name(Emp, Name)",
        # 7. Comparison only
        # "emp_salary(Emp, Salary), Salary = Salary",
        # 8. Variable not in any atom (should yield no results)
        # "Salary > 100000",
        # 9. Query with constants only
        # "emp_name('e_bob', 'bob')"
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
