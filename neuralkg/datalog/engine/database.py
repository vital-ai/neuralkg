from typing import Any
from ..model.schema import RelationSchema
from .frame import Frame, make_frame
from ..model.rule import Rule
from ..model.literal import Literal
from ..model.terms import Variable, Constant

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class DatalogDatabase:
    """
    Holds:
      - _relations: Dict[predicate -> (RelationSchema, Frame)]
      - rules: List[Rule]
    Provides methods to declare relations, add facts/rules, query, etc.
    Also provides API methods to load Datalog from files/strings and query from strings.
    """
    def __init__(self) -> None:
        self._relations: dict[str, tuple[RelationSchema, Frame]] = {}
        self.rules: list[Rule] = []

    def load_program_from_file(self, filepath: str) -> None:
        """
        Load a Datalog program (facts/rules) from a file and add to the database.
        """
        with open(filepath, 'r') as f:
            program = f.read()
        self.load_program_from_string(program)

    def load_program_from_string(self, program_str: str) -> None:
        """
        Load a Datalog program (facts/rules) from a string and add to the database.
        """
        from neuralkg.datalog.parser.datalog_parser import DatalogParser
        parser = DatalogParser()
        ast = parser.parse(program_str)
        self._load_from_ast(ast)

    def query_from_string(self, query_str: str) -> Frame:
        """
        Parse a query string and execute it against the database.
        Supports single atoms, conjunctions, negations, and comparisons.
        """
        # Parse the query string using DatalogParser
        from neuralkg.datalog.parser.datalog_parser import DatalogParser
        parser = DatalogParser()
        if not query_str.endswith('.'):
            query_str += '.'
        
        # Parse the query
        ast = parser.parse(query_str)
        logger.debug(f"[QUERY_FROM_STRING] AST: {ast}")
        
        # Extract variables from the AST
        user_var_order = []
        
        # Helper function to extract variables
        def extract_variables(node):
            """Extract variables from an AST node and its children."""
            vars_found = []
            # Handle dictionaries (typical AST nodes)
            if isinstance(node, dict):
                # Extract variables from atom arguments
                if node.get('type') == 'atom' and 'args' in node:
                    for arg in node.get('args', []):
                        if isinstance(arg, dict) and arg.get('type') == 'variable':
                            var_name = arg.get('name')
                            if var_name:
                                vars_found.append(var_name)
                # Extract variables from comparisons
                elif node.get('type') == 'comparison':
                    for side in ['left', 'right']:
                        if side in node:
                            # Handle both dict and Token objects
                            side_node = node[side]
                            if isinstance(side_node, dict) and side_node.get('type') == 'variable':
                                var_name = side_node.get('name')
                                if var_name:
                                    vars_found.append(var_name)
                            # If it's a Token object that might represent a variable
                            elif hasattr(side_node, 'type') and side_node.type == 'VARIABLE':
                                var_name = getattr(side_node, 'value', None)
                                if var_name:
                                    vars_found.append(var_name)
                # Extract variable directly if this is a variable node
                elif node.get('type') == 'variable':
                    var_name = node.get('name')
                    if var_name:
                        vars_found.append(var_name)
                # Process children recursively
                for key, value in node.items():
                    if key not in ['type', 'name']:
                        vars_found.extend(extract_variables(value))
            # Handle lists of nodes
            elif isinstance(node, list):
                for item in node:
                    vars_found.extend(extract_variables(item))
            # Handle Token objects
            elif hasattr(node, 'type') and node.type == 'VARIABLE':
                var_name = getattr(node, 'value', None)
                if var_name:
                    vars_found.append(var_name)
            return vars_found
            
        # Extract all variables from the AST
        all_vars = extract_variables(ast)
        # Remove duplicates while preserving order
        for var in all_vars:
            if var and var not in user_var_order:
                user_var_order.append(var)
                
        logger.debug(f"[QUERY_FROM_STRING] Extracted variables: {user_var_order}")
        
        # Process the head atom
        from ..model.literal import Literal
        from ..model.terms import Variable
        
        # Create head terms using variable order
        head_terms = [Variable(v) for v in user_var_order]
        
        # Special handling for __query__ relation - recreate for each query
        query_pred = "__query__"
        if query_pred in self._relations:
            del self._relations[query_pred]
            
        # Create synthetic head predicate with all variables
        head = Literal(query_pred, head_terms)
        
        # Extract the body parts from the query AST
        body = []
        comparisons = []
        
        # Helper function to extract body parts from the query AST
        def extract_body_parts(node):
            nonlocal body, comparisons
            if isinstance(node, dict):
                if node.get('type') == 'atom':
                    body.append(self._ast_atom_to_literal(node))
                elif node.get('type') == 'negation':
                    body.append(self._ast_atom_to_literal(node.get('atom', {}), negated=True))
                elif node.get('type') == 'comparison':
                    left = self._ast_term_to_model(node.get('left', {}))
                    right = self._ast_term_to_model(node.get('right', {}))
                    from ..model.literal import Comparison
                    comparisons.append(Comparison(left, node.get('op', '='), right))
                # Process children
                for key, value in node.items():
                    if key not in ['type', 'left', 'right']:
                        extract_body_parts(value)
            elif isinstance(node, list):
                for item in node:
                    extract_body_parts(item)
        
        # Extract body parts from the AST
        extract_body_parts(ast)
        
        # Add the temporary rule for the query
        rule = self.make_rule(head, body, comparisons)
        self.add_rule(rule)
        
        # Re-run full bottom-up evaluation (fixpoint) including the new rule
        from .evaluator import BottomUpEvaluator
        max_iterations = 10  # You can make this configurable if desired
        evaluator = BottomUpEvaluator(self, max_iterations=max_iterations)
        evaluator.evaluate()
        
        # Get the results
        result = None
        if query_pred in self._relations:
            result = self._relations[query_pred][1]
            
            # If we have results, rename columns to match the user variable names
            if result and result.num_rows() > 0:
                # The query relation has generic column names (arg0, arg1, etc.)
                # We want to rename them to match the user variable names
                rename_map = {}
                for i, var_name in enumerate(user_var_order):
                    rename_map[f"arg{i}"] = var_name
                
                # Rename the columns
                result = result.rename(rename_map)
            
            # Remove the synthetic query rule to avoid cluttering the database
            self.rules = [r for r in self.rules if r.head.predicate != query_pred]
            
            # Return the result frame
            return result
        
        # If no results were found, return an empty frame
        from ..engine.frame import make_frame
        return make_frame(columns=user_var_order)

    def _synthesize_rule_from_query(self, ast, user_var_order=None, allow_duplicates=False) -> str:
        """
        Given an AST for a conjunction/negation/comparison query, synthesize a Datalog rule string
        with a synthetic head predicate and all variables in the body as arguments.
        """
        import re
        def extract_vars(node):
            vars = set()
            if isinstance(node, dict):
                if node.get('type') == 'atom':
                    for t in node.get('terms', []):
                        vars |= extract_vars(t)
                elif node.get('type') == 'negation':
                    vars |= extract_vars(node['atom'])
                elif node.get('type') == 'comparison':
                    vars |= extract_vars(node['left'])
                    vars |= extract_vars(node['right'])
                elif node.get('type') == 'variable':
                    vars.add(node['name'])
                elif node.get('type') == 'number' or node.get('type') == 'constant' or node.get('type') == 'string':
                    pass
            elif isinstance(node, list):
                for item in node:
                    vars |= extract_vars(item)
            return vars

        # The body can be a list of atoms/negations/comparisons or a single dict
        body_items = ast if isinstance(ast, list) else [ast]
        vars_in_body = set()
        for item in body_items:
            vars_in_body |= extract_vars(item)
        # ALWAYS use user_var_order for the head, in order and with duplicates if present
        if user_var_order is not None and len(user_var_order) > 0:
            head_vars = user_var_order
        else:
            head_vars = sorted(vars_in_body)
        head = f"__query__({', '.join(head_vars)})" if head_vars else "__query__()"
        import json
        # Synthesize the body string
        def ast_to_str(node):
            if isinstance(node, dict):
                t = node.get('type')
                if t == 'atom':
                    args = ', '.join(ast_to_str(arg) for arg in node['terms'])
                    return f"{node['name']}({args})"
                elif t == 'negation':
                    return f"not {ast_to_str(node['atom'])}"
                elif t == 'comparison':
                    return f"{ast_to_str(node['left'])} {node['op']} {ast_to_str(node['right'])}"
                elif t == 'variable':
                    return node['name']
                elif t == 'number' or t == 'constant' or t == 'string':
                    return str(node['value']) if 'value' in node else str(node)
            elif isinstance(node, list):
                return ', '.join(ast_to_str(item) for item in node)
            return str(node)

        body_str = ast_to_str(ast)
        return f"{head} :- {body_str}."

    def _load_from_ast(self, ast):
        # Accepts AST as returned by DatalogParser
        stmts = []
        if isinstance(ast, dict):
            if 'clauses' in ast:
                stmts = ast['clauses']
            elif 'statements' in ast:
                stmts = ast['statements']
            elif ast.get('type') == 'program' and 'body' in ast:
                stmts = ast['body']
            else:
                stmts = [ast]
        elif isinstance(ast, list):
            stmts = ast
        else:
            raise ValueError("Unrecognized AST structure")
        def _flatten_body(body_elem, body_acc, comp_acc):
            """Recursively flatten body elements into body literals and comparisons."""
            if isinstance(body_elem, list):
                for be in body_elem:
                    _flatten_body(be, body_acc, comp_acc)
                return
            if not isinstance(body_elem, dict):
                return
            t = body_elem.get('type', None)
            if t == 'atom':
                body_acc.append(self._ast_atom_to_literal(body_elem))
            elif t == 'negation':
                body_acc.append(self._ast_atom_to_literal(body_elem['atom'], negated=True))
            elif t == 'comparison':
                left = self._ast_term_to_model(body_elem['left'])
                right = self._ast_term_to_model(body_elem['right'])
                from ..model.literal import Comparison
                comp_acc.append(Comparison(left, body_elem['op'], right))
            elif t == 'conjunction':
                # Defensive: flatten conjunctions
                _flatten_body(body_elem.get('body', []), body_acc, comp_acc)
            # else: ignore

        for stmt in stmts:
            if stmt['type'] == 'fact':
                lit = self._ast_atom_to_literal(stmt['atom'])
                # Do NOT call create_relation here; add_fact will create if needed, and will never overwrite existing data
                self.add_fact(lit)
                # Log schema and sample data after fact load
                try:
                    sch, fr = self._relations[lit.predicate]
                    logger.debug(f"[EDB_LOAD][FACT] Predicate '{lit.predicate}': schema={sch.colnames}, sample={fr.to_records()[:5] if hasattr(fr, 'to_records') else str(fr)[:300]}")
                except Exception as e:
                    logger.warning(f"[EDB_LOAD][FACT] Failed to log schema/data for '{lit.predicate}': {e}")
            elif stmt['type'] == 'rule':
                logger.debug(f"Rule stmt: {stmt}")  # Diagnostic
                if stmt['head']['name'] == 'path':
                    logger.debug(f"Rule stmt for path: {stmt}")  # Diagnostic
                head = self._ast_atom_to_literal(stmt['head'])
                body = []
                comparisons = []
                _flatten_body(stmt['body'], body, comparisons)
                self.add_rule(self.make_rule(head, body, comparisons))
        # Diagnostic print after all rules are loaded
        for rule in self.rules:
            logger.debug(f"Rule: {rule.head.predicate}({', '.join(str(t) for t in rule.head.terms)})")
            logger.debug(f"  Body literals: {[str(l) for l in rule.body_literals]}")
            logger.debug(f"  Comparisons: {[str(c) for c in rule.comparisons]}")
            logger.debug("")

    def _ast_term_to_model(self, term):
        # Accepts a term AST node and returns Variable or Constant
        # Handle Lark Token directly
        try:
            from lark.lexer import Token
        except ImportError:
            Token = None
        if Token is not None and isinstance(term, Token):
            if term.type == "VARIABLE":
                from ..model.terms import Variable
                return Variable(str(term))
            elif term.type == "CONSTANT":
                from ..model.terms import Constant
                return Constant(str(term))
            elif term.type == "NUMBER":
                from ..model.terms import Constant
                return Constant(int(term))
            elif term.type == "STRING":
                from ..model.terms import Constant
                # Remove quotes and unescape
                s = term[1:-1].encode("utf-8").decode("unicode_escape")
                return Constant(s)
        if isinstance(term, dict):
            if term['type'] == 'variable':
                from ..model.terms import Variable
                return Variable(term['name'])
            elif term['type'] == 'number':
                from ..model.terms import Constant
                return Constant(term['value'])
            elif term['type'] == 'string':
                from ..model.terms import Constant
                return Constant(term['value'])
            elif term['type'] == 'constant':
                from ..model.terms import Constant
                return Constant(term['value'])
        # fallback
        from ..model.terms import Constant
        return Constant(str(term))

    def _ast_atom_to_literal(self, atom, negated=False):
        # Accepts an atom AST node and returns a Literal
        from ..model.literal import Literal
        name = atom['name']
        terms = tuple(self._ast_term_to_model(t) for t in atom['terms'])
        return Literal(name, terms, negated=negated)

    def make_rule(self, head, body, comparisons):
        # Construct a Rule object from head, body_literals, and comparisons
        from ..model.rule import Rule
        return Rule(head, body_literals=body, comparisons=comparisons)

    def create_relation(self, predicate: str, arity: int, colnames: tuple[str, ...] = None) -> None:
        """
        Ensure a relation named `predicate` of the specified `arity` exists.
        If it already exists, verify the arity matches. Otherwise, create an
        empty Frame with the appropriate column names (defaults to arg* for facts, or variable names for rules).
        """
        logger.debug(f"[DIAG] create_relation called: predicate={predicate}, arity={arity}, colnames={colnames}")
        
        if predicate in self._relations:
            schema, frame = self._relations[predicate]
            if schema.arity != arity:
                # Special case for __query__ predicate - recreate it with new arity
                if predicate == "__query__":
                    logger.debug(f"[CREATE_RELATION] Recreating {predicate} with new arity {arity}")
                    del self._relations[predicate]
                else:
                    raise ValueError(
                        f"create_relation: '{predicate}' exists with arity {schema.arity}, not {arity}."
                    )
            else:
                # For existing relations with matching arity, we use the existing schema
                return
                
        if colnames is None:
            # Default column names: arg0, arg1, ..., arg{arity-1}
            colnames = tuple(f"arg{i}" for i in range(arity))
        schema = RelationSchema(predicate=predicate, arity=arity, colnames=colnames)
        empty_frame = make_frame.empty(list(colnames))
        self._relations[predicate] = (schema, empty_frame)

    def add_fact(self, atom: Literal) -> None:
        """
        Insert a ground fact (must be non-negated and ground) into the corresponding relation.
        E.g. `add_fact(Literal("edge",(Constant("a"),Constant("b")),negated=False))`.
        If the relation does not exist, create it. Avoid duplicates.
        """
        if atom.negated:
            raise ValueError("Cannot add a negative literal as a fact.")

        if not atom.is_ground():
            raise ValueError(f"Fact must be ground; got {atom}.")

        pred = atom.predicate
        arity = atom.arity()
        self.create_relation(pred, arity)

        schema, frame = self._relations[pred]
        # Patch: If the schema's colnames are not default argN, use their names by position
        default_argn = tuple(f"arg{i}" for i in range(arity))
        if tuple(schema.colnames) != default_argn:
            logger.debug(f"[ADD_FACT] Mapping fact for '{pred}' to custom schema columns {schema.colnames} from default {default_argn}")
        logger.debug(f"[ADD_FACT] Inserting fact for '{pred}' with schema columns {schema.colnames} and values {[atom.terms[i].value for i in range(arity)]}")
        # Always map fact values to schema columns by position, regardless of term names
        row = {schema.colnames[i]: atom.terms[i].value for i in range(arity)}
        logger.debug(f"[ADD_FACT][MAPPING] For predicate '{pred}', mapping fact values {[atom.terms[i].value for i in range(arity)]} to columns {schema.colnames}")
        candidate = make_frame.from_dicts([row], list(schema.colnames))
        logger.debug(f"[ADD_FACT] After insertion, columns for '{pred}': {frame.columns()}, sample row: {frame.to_records()[:1]}")

        if frame.num_rows() > 0:
            merged = frame.merge(candidate, how="inner",
                                 left_on=list(schema.colnames),
                                 right_on=list(schema.colnames))
            if merged.num_rows() > 0:
                return  # duplicate found

        appended = frame.concat([candidate])
        new_full = appended.drop_duplicates()
        self._relations[pred] = (schema, new_full)

    def get_relation(self, predicate: str) -> Frame:
        """
        Return a *copy* of the Frame for the given predicate. Raises KeyError if missing.
        """
        if predicate not in self._relations:
            raise KeyError(f"get_relation: no relation named '{predicate}'.")
        schema, frame = self._relations[predicate]
        return frame.copy()

    def relation_arity(self, predicate: str) -> int:
        """
        Return the arity of a given predicate. Raises KeyError if missing.
        """
        if predicate not in self._relations:
            raise KeyError(f"relation_arity: no relation named '{predicate}'.")
        schema, _ = self._relations[predicate]
        return schema.arity

    def add_rule(self, rule: Rule) -> None:
        """
        Add a rule (IDB). Ensures the relation for the head predicate exists with correct schema columns.
        Also synchronizes EDB schemas to variable names used in this rule's body.
        """
        head_pred = rule.head.predicate
        arity = rule.head.arity()
        colnames = tuple(
            t.name if isinstance(t, Variable) else f"arg{i}"
            for i, t in enumerate(rule.head.terms)
        )
        self.create_relation(head_pred, arity, colnames)
        for lit in getattr(rule, 'body_literals', []):
            pred = lit.predicate
            terms = lit.terms
            # Only synchronize for positive (non-negated) literals
            if pred in self._relations and not getattr(lit, 'negated', False):
                schema, _ = self._relations[pred]
                varnames = tuple(
                    t.name if isinstance(t, Variable) else schema.colnames[i]
                    for i, t in enumerate(terms)
                )
        try:
            if head_pred in self._relations:
                sch, fr = self._relations[head_pred]
                logger.debug(f"[RULE_ADD][HEAD] Predicate '{head_pred}': schema={sch.colnames}, sample={fr.to_records()[:5] if hasattr(fr, 'to_records') else str(fr)[:300]}")
            for lit in getattr(rule, 'body_literals', []):
                pred = lit.predicate
                if pred in self._relations:
                    sch, fr = self._relations[pred]
                    logger.debug(f"[RULE_ADD][BODY] Predicate '{pred}': schema={sch.colnames}, sample={fr.to_records()[:5] if hasattr(fr, 'to_records') else str(fr)[:300]}")
        except Exception as e:
            logger.warning(f"[RULE_ADD][SCHEMA_LOG] Failed to log schema/data: {e}")
        self.rules.append(rule)

    def all_predicates(self) -> list[str]:
        """
        Return the set of all predicates (from existing EDB relations and from rule heads/bodies).
        """
        preds = set(self._relations.keys())
        for r in self.rules:
            preds.add(r.head.predicate)
            for lit in r.body_literals:
                preds.add(lit.predicate)
        return list(preds)

    def query(self, goal: Literal) -> Frame:
        """
        Once the database is at fixpoint, retrieve all tuples satisfying `goal`.
        If `goal` is ground (no variables), returns a 0- or 1-row Frame.
        If `goal` has variables, returns a Frame whose columns are the variable names.
        """
        if goal.negated:
            raise ValueError("Cannot query a negated literal directly.")

        pred = goal.predicate
        if pred not in self._relations:
            # No such relation: no results
            return make_frame.empty([])

        schema, frame = self._relations[pred]
        df = frame.copy()

        # 1) Filter on any constant positions in goal
        for i, t in enumerate(goal.terms):
            match t:
                case Constant() as c:
                    df = df.filter_equals(schema.colnames[i], c.value)
                case Variable():
                    continue
                case _:
                    raise ValueError(f"Unexpected term in query: {t}")

        if goal.is_ground():
            return df  # 0 or 1 row

        # Project columns by position for variable queries (Datalog semantics)
        var_positions = [(i, t.name) for i, t in enumerate(goal.terms) if isinstance(t, Variable)]
        colnames = list(schema.colnames)
        records = df.to_records()
        projected = [
            {var_name: list(row.values())[i] for i, var_name in enumerate([v for _, v in var_positions])}
            for row in records
        ]
        return make_frame.from_dicts(projected, [v for _, v in var_positions])

    def reset(self) -> None:
        """
        Clear all stored facts (empty all Frames) and drop all rules.
        Schemas remain declared.
        """
        for pred, (schema, _) in self._relations.items():
            empty_frame = make_frame.empty(list(schema.colnames))
            self._relations[pred] = (schema, empty_frame)
        self.rules.clear()
