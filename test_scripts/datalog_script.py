import sys
import os


from neuralkg.datalog.engine.evaluator import BottomUpEvaluator
print("ACTUAL BottomUpEvaluator file:", sys.modules[BottomUpEvaluator.__module__].__file__)
print("PYTHON EXECUTABLE:", sys.executable)
with open(sys.modules[BottomUpEvaluator.__module__].__file__) as f:
    print("----- FILE CONTENTS -----")
    for i, line in enumerate(f):
        if i > 20: break
        print(line.rstrip())
    print("-------------------------")
print("START OF SCRIPT datalog_script.py")
import pandas as pd
from neuralkg.datalog.engine.frame import make_frame
from neuralkg.datalog.engine.database import DatalogDatabase
from neuralkg.datalog.model.terms import Variable, Constant
from neuralkg.datalog.model.literal import Literal, AggregateSpec
from neuralkg.datalog.model.rule import Rule


def main():
    print('ENTERED MAIN (extended aggregates)')
    db = DatalogDatabase()
    # --- Example: Aggregates (including the new ones: first, last, product, mode, string_agg) ---
    # Facts (EDB):
    db.create_relation("purchase", 3)
    db.add_fact(Literal("purchase", (Constant("john"),  Constant("order1"), Constant(100)), negated=False))
    db.add_fact(Literal("purchase", (Constant("john"),  Constant("order2"), Constant(200)), negated=False))
    db.add_fact(Literal("purchase", (Constant("john"),  Constant("order3"), Constant(200)), negated=False))
    db.add_fact(Literal("purchase", (Constant("mary"),  Constant("order4"), Constant(150)), negated=False))
    db.add_fact(Literal("purchase", (Constant("mary"),  Constant("order5"), Constant(50)),  negated=False))

    # 2) Variables
    C     = Variable("C")       # Customer
    O     = Variable("O")       # Order ID
    Amt   = Variable("Amt")     # Amount

    SumVal   = Variable("SumVal")
    MinVal   = Variable("MinVal")
    MaxVal   = Variable("MaxVal")
    MedVal   = Variable("MedVal")
    StdVal   = Variable("StdVal")
    VarVal   = Variable("VarVal")
    NumOrder = Variable("NumOrder")
    Lst      = Variable("Lst")
    FO       = Variable("FO")
    LO       = Variable("LO")
    ProdVal  = Variable("ProdVal")
    ModeVal  = Variable("ModeVal")
    StrVal   = Variable("StrVal")

    # 3) Define each aggregate rule:
    agg_sum = AggregateSpec(func="sum", group_by=(C,), target=Amt, result=SumVal)
    rule_sum = Rule(
        head=Literal("total_spent", (C, SumVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_sum
    )
    db.add_rule(rule_sum)

    agg_min = AggregateSpec(func="min", group_by=(C,), target=Amt, result=MinVal)
    rule_min = Rule(
        head=Literal("min_spent", (C, MinVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_min
    )
    db.add_rule(rule_min)

    agg_max = AggregateSpec(func="max", group_by=(C,), target=Amt, result=MaxVal)
    rule_max = Rule(
        head=Literal("max_spent", (C, MaxVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_max
    )
    db.add_rule(rule_max)

    agg_med = AggregateSpec(func="median", group_by=(C,), target=Amt, result=MedVal)
    rule_med = Rule(
        head=Literal("median_spent", (C, MedVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_med
    )
    db.add_rule(rule_med)

    agg_std = AggregateSpec(func="std", group_by=(C,), target=Amt, result=StdVal)
    rule_std = Rule(
        head=Literal("std_spent", (C, StdVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_std
    )
    db.add_rule(rule_std)

    agg_var = AggregateSpec(func="var", group_by=(C,), target=Amt, result=VarVal)
    rule_var = Rule(
        head=Literal("var_spent", (C, VarVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_var
    )
    db.add_rule(rule_var)

    agg_cd = AggregateSpec(func="count_distinct", group_by=(C,), target=O, result=NumOrder)
    rule_cd = Rule(
        head=Literal("distinct_orders", (C, NumOrder), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_cd
    )
    db.add_rule(rule_cd)

    agg_coll = AggregateSpec(func="collect", group_by=(C,), target=O, result=Lst)
    rule_coll = Rule(
        head=Literal("collect_orders", (C, Lst), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_coll
    )
    db.add_rule(rule_coll)

    agg_first = AggregateSpec(func="first", group_by=(C,), target=O, result=FO)
    rule_first = Rule(
        head=Literal("first_order", (C, FO), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_first
    )
    db.add_rule(rule_first)

    agg_last = AggregateSpec(func="last", group_by=(C,), target=O, result=LO)
    rule_last = Rule(
        head=Literal("last_order", (C, LO), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_last
    )
    db.add_rule(rule_last)

    agg_prod = AggregateSpec(func="product", group_by=(C,), target=Amt, result=ProdVal)
    rule_prod = Rule(
        head=Literal("product_spent", (C, ProdVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_prod
    )
    db.add_rule(rule_prod)

    agg_mode = AggregateSpec(func="mode", group_by=(C,), target=Amt, result=ModeVal)
    rule_mode = Rule(
        head=Literal("mode_spent", (C, ModeVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_mode
    )
    db.add_rule(rule_mode)

    agg_str = AggregateSpec(func="string_agg", group_by=(C,), target=O, result=StrVal)
    rule_str = Rule(
        head=Literal("string_orders", (C, StrVal), negated=False),
        body_literals=[Literal("purchase", (C, O, Amt), negated=False)],
        comparisons=[],
        aggregate=agg_str
    )
    db.add_rule(rule_str)

    # 4) Run evaluation
    evaluator = BottomUpEvaluator(db)
    print("[DEBUG] evaluator type:", type(evaluator), evaluator.__class__.__module__)
    import sys
    print("[DEBUG] evaluator module file:", sys.modules[BottomUpEvaluator.__module__].__file__)
    print("CALLING EVALUATE: extended aggregates")
    evaluator.evaluate()

    # 5) Inspect results
    print("---- Extended Aggregation Example (all functions) ----\n")
    for pred in [
        "total_spent", "min_spent", "max_spent", "median_spent",
        "std_spent", "var_spent", "distinct_orders", "collect_orders",
        "first_order", "last_order", "product_spent", "mode_spent", "string_orders"
    ]:
        df = db.get_relation(pred)
        print(f"Predicate `{pred}`:")
        for row in df.to_records():
            print(" ", row)
        print()

    # 6) Sample Queries:
    #    What is the total spent by "john"?
    goal_sum_john = Literal("total_spent", (Constant("john"), SumVal))
    print("Query: total_spent(\"john\", X) →", db.query(goal_sum_john).to_records())

    #    What is the first order of "mary"?
    goal_fo_mary = Literal("first_order", (Constant("mary"), FO))
    print("Query: first_order(\"mary\", X) →", db.query(goal_fo_mary).to_records())

    #    What is the mode of amounts spent by "john"?
    goal_mode_john = Literal("mode_spent", (Constant("john"), ModeVal))
    print("Query: mode_spent(\"john\", X) →", db.query(goal_mode_john).to_records())

    #    What is the concatenation of orders for "john"?
    goal_str_john = Literal("string_orders", (Constant("john"), StrVal))
    print("Query: string_orders(\"john\", X) →", db.query(goal_str_john).to_records())

    print("\n---- Done ----")

if __name__ == "__main__":
    main()
