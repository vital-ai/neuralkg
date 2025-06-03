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
        from neuralkg.datalog.parser.datalog_parser import DatalogParser
        parser = DatalogParser()
        if not query_str.endswith('.'):
            query_str += '.'
        import re
        single_atom_pattern = re.compile(r'^\s*[a-zA-Z_][a-zA-Z0-9_]*\s*\(([^()]*)\)\s*\.?\s*$')
        if single_atom_pattern.match(query_str.strip()):
            # Try to parse as a single atom/fact
            ast = parser.parse(query_str)
            clause = ast
            if isinstance(ast, dict) and ast.get('type') == 'program':
                clauses = ast.get('clauses') or ast.get('statements') or []
                if clauses:
                    clause = clauses[0]
            if isinstance(clause, dict) and clause.get('type') == 'fact':
                literal = self._ast_atom_to_literal(clause['atom'])
                return self.query(literal)
            elif isinstance(clause, dict) and clause.get('type') == 'atom':
                literal = self._ast_atom_to_literal(clause)
                return self.query(literal)
        # Otherwise, synthesize a rule string for complex queries
        # Parse the query string as a query (conjunction/body) using the new grammar
        parser = DatalogParser()
        ast_query = parser.parse(query_str)
        def extract_var_order_from_query(node):
            order = []
            try:
                from lark.lexer import Token
            except ImportError:
                Token = None
            if Token is not None and isinstance(node, Token):
                if node.type == "VARIABLE":
                    order.append(str(node))
                return order
            if isinstance(node, dict):
                if node.get('type') == 'atom':
                    for t in node.get('terms', []):
                        order += extract_var_order_from_query(t)
                elif node.get('type') == 'negation':
                    order += extract_var_order_from_query(node['atom'])
                elif node.get('type') == 'comparison':
                    order += extract_var_order_from_query(node['left'])
                    order += extract_var_order_from_query(node['right'])
                elif node.get('type') == 'variable':
                    order.append(node['name'])
            elif isinstance(node, list):
                for item in node:
                    order += extract_var_order_from_query(item)
            return order
        var_order = []
        # Handle new 'query' AST type
        if isinstance(ast_query, dict) and ast_query.get('type') == 'query':
            var_order = extract_var_order_from_query(ast_query['body'])
        elif isinstance(ast_query, dict) and ast_query.get('type') == 'program':
            clauses = ast_query.get('clauses') or ast_query.get('statements') or []
            if clauses:
                ast_query = clauses[0]
            if isinstance(ast_query, dict) and ast_query.get('type') == 'query':
                var_order = extract_var_order_from_query(ast_query['body'])
        elif isinstance(ast_query, dict):
            if ast_query.get('type') == 'fact':
                ast_query = ast_query['atom']
            if ast_query.get('type') == 'atom':
                var_order = extract_var_order_from_query(ast_query)
            elif ast_query.get('type') == 'rule':
                body = ast_query['body'] if isinstance(ast_query['body'], list) else [ast_query['body']]
                for b in body:
                    var_order += extract_var_order_from_query(b)
        # Always extract user_var_order from the query body AST, regardless of type
        # Print the relevant AST node for diagnostics
        if isinstance(ast_query, dict) and ast_query.get('type') == 'query':
            query_body_ast = ast_query['body']
        else:
            query_body_ast = ast_query

        var_order = extract_var_order_from_query(query_body_ast)
        # Only keep unique variables in order of first appearance
        seen = set()
        user_var_order = []
        for v in var_order:
            if v not in seen:
                user_var_order.append(v)
                seen.add(v)

        # Guarantee: synthetic rule head uses all variables from user query, in order, with duplicates if present
        rule_ast = query_body_ast
        rule_str = self._synthesize_rule_from_query(rule_ast, user_var_order, allow_duplicates=True)
        parser = DatalogParser()
        ast = parser.parse(rule_str)
        # Defensive: if ast is a program, extract the first clause
        clause = ast
        if isinstance(ast, dict) and ast.get('type') == 'program':
            clauses = ast.get('clauses') or ast.get('statements') or []
            if not clauses:
                raise ValueError("Synthesized rule parsing produced empty program AST.")
            clause = clauses[0]
        if not (isinstance(clause, dict) and clause.get('type') == 'rule'):
            raise ValueError(f"Synthesized rule parsing did not yield a rule AST: {clause}")
        # Force the head to use user_var_order as variable names, if provided
        if user_var_order:
            from ..model.terms import Variable
            head_vars = tuple(Variable(name) for name in user_var_order)
            head = type(self._ast_atom_to_literal(clause['head']))('__query__', head_vars, negated=False)
        else:
            head = self._ast_atom_to_literal(clause['head'])
        body_elems = clause['body'] if isinstance(clause['body'], list) else [clause['body']]
        body = []
        comparisons = []
        for b in body_elems:
            if not isinstance(b, dict):
                continue  # skip stray tokens or strings
            if b['type'] == 'atom':
                body.append(self._ast_atom_to_literal(b))
            elif b['type'] == 'negation':
                body.append(self._ast_atom_to_literal(b['atom'], negated=True))
            elif b['type'] == 'comparison':
                left = self._ast_term_to_model(b['left'])
                right = self._ast_term_to_model(b['right'])
                from ..model.literal import Comparison
                comparisons.append(Comparison(left, b['op'], right))
        # Add the temporary rule for the query
        rule = self.make_rule(head, body, comparisons)
        self.add_rule(rule)
        head_pred = head.predicate
        # Save the current state of the synthetic predicate's relation (if any)
        prev_relation = None
        if head_pred in self._relations:
            prev_relation = self._relations[head_pred][1].copy()
        # Re-run full bottom-up evaluation (fixpoint) including the new rule
        from .evaluator import BottomUpEvaluator
        max_iterations = 10  # You can make this configurable if desired
        evaluator = BottomUpEvaluator(self, max_iterations=max_iterations)
        try:
            evaluator.evaluate()
        except RuntimeError as e:
            # Clean up before raising
            self.rules.pop()
            if prev_relation is not None:
                schema = self._relations[head_pred][0]
                self._relations[head_pred] = (schema, prev_relation)
            else:
                if head_pred in self._relations:
                    del self._relations[head_pred]
            raise RuntimeError(f"Query evaluation exceeded max iterations ({max_iterations}): {e}")
        # Query the synthetic predicate
        result = self.query(head)
        # No mapping/renaming needed: synthetic rule head uses user variable names/order
        # Remove the temporary rule
        self.rules.pop()
        # Restore the synthetic relation to its previous state
        if prev_relation is not None:
            schema = self._relations[head_pred][0]
            self._relations[head_pred] = (schema, prev_relation)
        else:
            if head_pred in self._relations:
                del self._relations[head_pred]
        logger.debug(f"[QUERY_FROM_STRING] user_var_order: {user_var_order}")
        logger.debug(f"[QUERY_FROM_STRING] Final result columns before projection: {result.columns()}")
        logger.debug(f"[QUERY_FROM_STRING] Final result sample rows: {result.to_records()[:5]}")
        # --- PATCH: Explicitly project and rename to user variable order ---
        # Always project and rename to user variable order for synthetic queries
        # Only project if user_var_order is non-empty (i.e., there are variables in the query)
        if user_var_order and len(user_var_order) > 0:
            result_cols = result.columns()
            logger.debug(f"[QUERY_FROM_STRING] user_var_order: {user_var_order}")
            logger.debug(f"[QUERY_FROM_STRING] result columns: {result_cols}")
            # If all user_var_order columns exist, project as usual
            if all(col in result_cols for col in user_var_order):
                records = result.to_records()
                projected = [{k: row.get(k, None) for k in user_var_order} for row in records]
                result = type(result).from_dicts(projected, user_var_order)
                logger.debug(f"[QUERY_FROM_STRING] Projected columns: {user_var_order}")
            # If not, but the number of columns matches, project by position
            elif len(result_cols) == len(user_var_order):
                logger.debug("[QUERY_FROM_STRING] Column names mismatch but arity matches; projecting by position.")
                records = result.to_records()
                projected = [dict(zip(user_var_order, [row[c] for c in result_cols])) for row in records]
                result = type(result).from_dicts(projected, user_var_order)
                logger.debug(f"[QUERY_FROM_STRING] Projected by position: {user_var_order}")
            # Otherwise, fallback to returning all columns
            else:
                logger.warning(f"[QUERY_FROM_STRING] Cannot project to user_var_order {user_var_order}, returning all columns: {result_cols}")
        logger.debug(f"[QUERY_FROM_STRING] Result columns after projection: {result.columns()}")
        logger.debug(f"[QUERY_FROM_STRING] Result sample rows after projection: {result.to_records()[:5]}")
        logger.debug(f"[QUERY_FROM_STRING] Result columns after projection: {result.columns()}")
        logger.debug(f"[QUERY_FROM_STRING] Result sample rows after projection: {result.to_records()[:5]}")
        return result

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
        logger.debug(f"[DIAG] create_relation called: predicate={predicate}, arity={arity}, colnames={colnames}")
        """
        Ensure a relation named `predicate` of the specified `arity` exists.
        If it already exists, verify the arity matches. Otherwise, create an
        empty Frame with the appropriate column names (defaults to arg* for facts, or variable names for rules).
        """
        if predicate in self._relations:
            schema, frame = self._relations[predicate]
            if schema.arity != arity:
                raise ValueError(
                    f"create_relation: '{predicate}' exists with arity {schema.arity}, not {arity}."
                )
            # Never overwrite the existing frame or schema
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
