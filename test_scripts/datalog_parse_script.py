from neuralkg.datalog.parser.datalog_parser import DatalogParser
import pprint
def main():

    with open("test_data/dl1_test.datalog", "r") as f:
        sample_program = f.read()

    parser = DatalogParser()

    ast = parser.parse(sample_program)
    
    pprint.pprint(ast)

    test_programs = [
        (
            "Simple fact and comparison",
            r"""
                a(X) :- X < 5.
            """,
        ),
        (
            "Multiple clauses",
            r"""
                edge(a, b, 50).
                path(X, Y) :- edge(X, Y, W), W >= 50.
            """,
        ),
    ]

    parser = DatalogParser()

    for desc, prog in test_programs:
        print(f"--- {desc} ---")
        try:
            ast = parser.parse(prog)
            pprint.pprint(ast)
        except Exception as e:
            print(f"Error parsing '{desc}': {e}")

    extra_tests = [
        (
            "Chained comparisons and nested facts",
            r"""
                employee(john).
                salary(john, 300).
                senior_employee(X) :- employee(X), salary(X, S), S > 200.
            """,
        ),
        (
            "No body rule (should fail)",
            r"""
                invalid_rule(X) :- .
            """,
        ),
    ]

    for desc, prog in extra_tests:
        print(f"--- {desc} ---")
        try:
            ast = parser.parse(prog)
            pprint.pprint(ast)
        except Exception as e:
            print(f"Error parsing '{desc}': {e}")
            
            

if __name__ == "__main__":
    main()




"""
-- 1. Get all “light edges” (i.e., edges with weight < 200):
? light_edge(X, Y).

-- 2. Get the full “edge3” info for light edges:
? light_edge3(X, Y, W).

-- 3. Get the full transitive closure over light edges:
? path(X, Y).

-- 4. Get all eligible employees (name only):
? eligible(Name).

-- 5. Get full info on eligible employees:
? eligible3(Name, Age, Salary).

-- 6. Reconstructed original graphs:
? edge3(X, Y, W).

-- 7. Reconstructed original employee facts:
? employee3(Name, Age, Salary).

-- 8. Combine graph & employee logic in more complex queries if needed.
*/
"""
