from problog.program     import PrologFile
from problog.engine      import DefaultEngine
from problog.sdd_formula import SDD
import json
from problog.formula import LogicFormula
import torch
# from klay       import Circuit


def main():
    # 1. Load and ground the ProbLog program
    engine = DefaultEngine()
    db = engine.prepare(PrologFile('problog_data.pl'))
    formula = engine.ground_all(db)

    # 2. Extract fully grounded query terms for mapping
    raw_qs = formula.queries()  # list of (Term, key)
    query_terms = [term for term, _ in raw_qs]
    query_strs = [str(term) for term in query_terms]
    print(f"Prepared {len(query_terms)} ground queries.")

    # 3. Compile multi-rooted SDD in memory
    sdd_wrap = SDD.create_from(formula, queries=query_terms)
    manager = sdd_wrap.inode_manager.get_manager()  # PySDD SddManager

    # 4. Build KLay circuit by adding each root node
    roots = manager.vtree().get_sdd_rootnodes(manager)
    
    """
    circ = Circuit()
    for root in roots:
        circ.add_sdd(root)
    circ.remove_unused_nodes()
    print(f"KLay loaded: {circ.nb_root_nodes()} roots, {circ.nb_nodes()} total nodes")

    # 5. Compile to a PyTorch KnowledgeModule expecting leaf weights
    module = circ.to_torch_module(semiring="real", probabilistic=True)
    module.eval()

    # 6. Extract literal weights via WMC from the first root
    #    Use the WmcManager returned by SddNode.wmc()
    wmc_mgr = roots[0].wmc(log_mode=False)  # pysdd.sdd.WmcManager
    num_vars = wmc_mgr.var_count()  # number of variables
    weights = torch.tensor([wmc_mgr.literal_weight(i) for i in range(1, num_vars + 1)], dtype=torch.float32)

    # 7. Evaluate all root probabilities in one pass Evaluate all root probabilities in one pass
    with torch.no_grad():
        outputs = module(weights)

    # 8. Lookup and print specific ground query
    target = 'conflict_intensity(brazil,brazil)'
    if target not in query_strs:
        raise KeyError(f"Query '{target}' not found among grounded queries")
    idx = query_strs.index(target)
    prob = outputs[idx].item()
    print(f"P({target}) = {prob:.6f}")
    """

if __name__ == '__main__':
    main()


