from problog.logic import Term
from problog.program import PrologFile
from problog.engine import DefaultEngine
from problog.sdd_formula import SDD
from problog import get_evaluatable

engine = DefaultEngine()

model = PrologFile('problog_data.pl')

db = engine.prepare(model)

sdd = SDD.create_from(db)

# print(sdd)

evaluator = get_evaluatable().create_from(db)

results = evaluator.evaluate()

def query_probability(functor: str, arg1: str, arg2: str) -> float:

    term = Term(functor, Term(arg1), Term(arg2))

    return results.get(term, 0.0)


def query_all(functor: str):
    """
    Return a list of (arg1, arg2, probability) by querying for non-ground functor(A,B).
    Uses engine.query to enumerate all ground bindings, then looks up probabilities.
    """
    # Build a non-ground query term with None for variables
    query_term = Term(functor, None, None)
    # engine.query returns a list of tuples of Terms for which the query succeeds
    bindings = engine.query(db, query_term)  # List[Tuple[Term, Term]] ([problog.readthedocs.io](https://problog.readthedocs.io/en/latest/_modules/problog/engine.html?utm_source=chatgpt.com))
    results_list = []
    for (a, b) in bindings:
        a_str = str(a)
        b_str = str(b)
        prob = query_probability(functor, a_str, b_str)
        results_list.append((a_str, b_str, prob))
    return results_list

def main():

    # 3. Ground all queries together (single grounding pass)
    # formula = engine.ground_all(db, queries=queries)

    # 4. Compile into a Sentential Decision Diagram (SDD)
    # sdd = SDD.create_from(formula)

    # 5. Evaluate the SDD (weighted model counting)
    # results = sdd.evaluate()

    test_queries = [
        ('accusation', 'brazil', 'brazil'),
        ('cultural_bridge', 'burma', 'israel'),
        ('conflict_intensity', 'usa', 'india'),
        ('hostile_exchange', 'russia', 'ukraine'),
        ('diplomatic_siege', 'china', 'india'),
        ('military_alignment', 'nato', 'russia'),
        ('humanitarian_concern', 'usa', 'haiti'),
        ('shared_indoctrination', 'china', 'nepal'),
    ]

    print("Testing ProbLog batch queries:")

    for functor, arg1, arg2 in test_queries:
        prob = query_probability(functor, arg1, arg2)
        print(f"{prob:.4f} :: {functor}({arg1}, {arg2})")

    print("Bindings and probabilities for conflict_intensity(A, B):")
    for a1, a2, p in query_all('conflict_intensity'):
        print(f"{p:.4f} :: conflict_intensity({a1}, {a2})")


if __name__ == '__main__':
    main()
