import logging
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Union, Optional, Any
from tqdm import tqdm

from ..model.schema import RelationSchema
from .frame import Frame
from .frame_factory import FrameFactory
from ..model.rule import Rule
from ..model.literal import Literal, Comparison
from ..model.terms import Variable, Constant
from ..parser.datalog_parser import DatalogParser

# Define aggregate function names here to avoid circular import
AGGREGATE_FUNCS = ['agg_sum', 'agg_avg', 'agg_count', 'agg_max', 'agg_min']

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

class QueryResult:
    """A wrapper for query results that provides a consistent interface regardless
    of whether the underlying frame is None, empty, or contains data.
    
    This ensures that calling code can always access methods like num_rows(), columns(),
    and to_records() without checking for None first.
    """
    def __init__(self, frame=None, status="success", message=""):
        self.frame = frame
        self.status = status  # success, empty, error
        self.message = message
    
    def is_empty(self) -> bool:
        """Returns True if the result is empty (no rows)"""
        return self.frame is None or self.frame.num_rows() == 0
    
    def has_error(self) -> bool:
        """Returns True if there was an error during query execution"""
        return self.status == "error"
    
    def num_rows(self) -> int:
        """Returns the number of rows in the result"""
        return 0 if self.frame is None else self.frame.num_rows()
    
    def columns(self) -> List[str]:
        """Returns the column names in the result"""
        return [] if self.frame is None else self.frame.columns()
    
    def to_records(self) -> List[Dict[str, Any]]:
        """Returns the result data as a list of dictionaries"""
        return [] if self.frame is None else self.frame.to_records()
    
    @classmethod
    def empty(cls, columns=None):
        """Create an empty result with the specified schema"""
        # Get reference to the self._make_frame function from outer DatalogDatabase class
        # This ensures we use the proper implementation even when used outside the database
        make_frame_func = DatalogDatabase._make_frame if hasattr(DatalogDatabase, '_make_frame') else None
        
        if columns:
            frame = make_frame_func.from_dicts([], columns=columns) if make_frame_func else FrameFactory.from_dicts([], columns=columns)
        else:
            frame = make_frame_func.from_dicts([], columns=[]) if make_frame_func else FrameFactory.from_dicts([], columns=[])
        return cls(frame=frame, status="empty", message="No results found")
    
    @classmethod
    def error(cls, message):
        """Create an error result with the specified message"""
        return cls(frame=None, status="error", message=message)

class DatalogDatabase:
    """
    Holds:
      - _relations: Dict[predicate -> (RelationSchema, Frame)]
      - rules: List[Rule]
    Provides methods to declare relations, add facts/rules, query, etc.
    Also provides API methods to load Datalog from files/strings and query from strings.
    
    Each database instance can have its own Frame implementation type, independent
    of other database instances or the global FrameFactory setting.
    """
    def __init__(self) -> None:
        from .frame_factory import FrameFactory
        
        # Store the implementation name and class for this database instance
        self._implementation_name = FrameFactory.get_current_implementation_name()
        self._frame_class = FrameFactory.get_implementation()
        
        # Store a reference to the make_frame function for this implementation
        # This ensures this database always uses its assigned implementation
        # regardless of global FrameFactory state changes
        self._make_frame = FrameFactory.get_make_frame_function()
        
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

    def _extract_variables(self, query_body):
        """Extract variables from a list of literals."""
        variables = []
        for lit in query_body:
            for term in lit.terms:
                if hasattr(term, 'name') and term not in variables:  # Variable objects have 'name' attribute
                    variables.append(term)
        return variables
        
    def _extract_variables_from_dict(self, node):
        """Extract all variables from a dictionary representation of a query AST node.
        
        Args:
            node: Dictionary representing an AST node
            
        Returns:
            Set of variable names found in the node
        """
        vars = set()
        if isinstance(node, dict):
            if node.get('type') == 'atom':
                for term in node.get('terms', []):
                    vars.update(self._extract_variables_from_dict(term))
            elif node.get('type') == 'negation':
                vars.update(self._extract_variables_from_dict(node.get('atom', {})))
            elif node.get('type') == 'comparison':
                vars.update(self._extract_variables_from_dict(node.get('left', {})))
                vars.update(self._extract_variables_from_dict(node.get('right', {})))
            elif node.get('type') == 'variable':
                vars.add(node.get('name'))
        elif isinstance(node, list):
            for item in node:
                vars.update(self._extract_variables_from_dict(item))
        return vars
        
    def _convert_dict_to_literal(self, node):
        """Convert a dictionary representation of a literal to a Literal or Comparison object.
        
        Args:
            node: Dictionary representing a literal AST node or a Lark Token
            
        Returns:
            Literal/Comparison object or None if conversion fails
        """
        from ..model.literal import Literal, Comparison
        from ..model.terms import Variable, Constant
        
        # Try to handle Lark.Token objects directly
        try:
            from lark.lexer import Token
            if isinstance(node, Token):
                if node.type == "VARIABLE":
                    return Variable(str(node))
                elif node.type == "CONSTANT":
                    return Constant(str(node))
                elif node.type == "NUMBER":
                    return Constant(int(node))
                elif node.type == "STRING":
                    # Remove quotes and unescape
                    s = str(node)[1:-1].encode("utf-8").decode("unicode_escape")
                    return Constant(s)
        except ImportError:
            pass
            
        if isinstance(node, dict):
            # Handle regular atom
            if node.get('type') == 'atom':
                predicate = node.get('name')
                terms = []
                for term in node.get('terms', []):
                    # Convert each term recursively to handle nested structures
                    term_obj = self._ast_term_to_model(term)
                    if term_obj is not None:
                        terms.append(term_obj)
                return Literal(predicate, terms)
            # Handle negation
            elif node.get('type') == 'negation':
                atom_lit = self._convert_dict_to_literal(node.get('atom', {}))
                if atom_lit and isinstance(atom_lit, Literal):
                    # Can only negate Literals, not Comparisons
                    return Literal(atom_lit.predicate, atom_lit.terms, negated=True)
                return atom_lit
            # Handle comparison (create a Comparison object)
            elif node.get('type') == 'comparison':
                op = node.get('op')
                left = node.get('left', {})
                right = node.get('right', {})
                
                # Convert left and right to terms
                left_term = self._ast_term_to_model(left)
                right_term = self._ast_term_to_model(right)
                    
                if left_term and right_term:
                    # Create a Comparison object
                    return Comparison(left_term, op, right_term)
        return None
        
    def _ast_term_to_model(self, term):
        """Accepts a term AST node or Lark Token and returns Variable or Constant"""
        from ..model.terms import Variable, Constant
        
        # Handle Lark Token directly
        try:
            from lark.lexer import Token
            if isinstance(term, Token):
                if term.type == "VARIABLE":
                    return Variable(str(term))
                elif term.type == "CONSTANT":
                    return Constant(str(term))
                elif term.type == "NUMBER":
                    return Constant(int(term))
                elif term.type == "STRING":
                    # Remove quotes and unescape
                    s = str(term)[1:-1].encode("utf-8").decode("unicode_escape")
                    return Constant(s)
        except (ImportError, AttributeError):
            pass
            
        if isinstance(term, dict):
            if term.get('type') == 'variable':
                return Variable(term.get('name'))
            elif term.get('type') == 'number':
                return Constant(term.get('value'))
            elif term.get('type') == 'string':
                return Constant(term.get('value'))
            elif term.get('type') == 'constant':
                return Constant(term.get('value'))
        
        # Fallback
        try:
            return Constant(str(term))
        except:
            return None

    def query_from_string(self, query_str):
        """Execute a query from a string.

        Args:
            query_str: A Datalog query string.

        Returns:
            A QueryResult object containing the result of the query.
        """
        try:
            logger.debug(f"[QUERY_STRING] Parsing query: {query_str}")
            parser = DatalogParser()
            query_result = parser.parse(query_str)
            
            # Handle case where parse result might be None
            if not query_result:
                logger.error(f"[QUERY_STRING] Failed to parse query: {query_str}")
                return QueryResult.error(f"Failed to parse query: {query_str}")
                
            # Handle various query result structures from the parser
            if isinstance(query_result, dict):
                # Check for program structure with clauses (common for single-predicate queries)
                if query_result.get('type') == 'program' and 'clauses' in query_result:
                    # For single query atoms like 'light_edge(X, Y).'
                    clauses = query_result.get('clauses', [])
                    if len(clauses) == 1 and clauses[0].get('type') == 'fact':
                        atom = clauses[0].get('atom')
                        if atom:
                            query_body = [atom]
                        else:
                            logger.error(f"[QUERY_STRING] Invalid atom in program structure: {query_result}")
                            return QueryResult.error(f"Invalid atom in query: {query_str}")
                    else:
                        # Multiple clauses or not a fact - unexpected in a query
                        logger.error(f"[QUERY_STRING] Unexpected clauses in program structure: {clauses}")
                        return QueryResult.error(f"Invalid query structure: {query_str}")
                # Direct atom structure
                elif query_result.get('type') == 'atom':
                    query_body = [query_result]
                # Complex query with explicit body field
                elif 'body' in query_result:
                    query_body = query_result.get('body', [])
                else:
                    logger.error(f"[QUERY_STRING] Unexpected dictionary structure: {query_result}")
                    return QueryResult.error(f"Invalid query structure: {query_str}")
            # List of atoms (comma-separated predicates)
            elif isinstance(query_result, list):
                query_body = query_result
            else:
                logger.error(f"[QUERY_STRING] Unexpected query structure: {query_result}")
                return QueryResult.error(f"Invalid query structure: {query_str}")
                
            logger.debug(f"[QUERY_STRING] Parsed query body: {query_body}")
            
            # Create a synthetic head predicate and construct a rule with the body
            # Extract all variables from the query body
            variables = set()
            for lit in query_body:
                if isinstance(lit, dict):
                    extracted_vars = self._extract_variables_from_dict(lit)
                    variables.update(extracted_vars)
                    
            # Convert query_body dictionary into a list of proper Literal objects
            body_literals = []
            comparisons = []
            
            for lit in query_body:
                if isinstance(lit, dict):
                    # Convert dictionary literals to proper Literal objects
                    body_lit = self._convert_dict_to_literal(lit)
                    if body_lit:
                        if isinstance(body_lit, Comparison):
                            comparisons.append(body_lit)
                        else:
                            body_literals.append(body_lit)
            
            # Check if this is an aggregate query
            has_aggregates = False
            agg_result_vars = []
            for lit in body_literals:
                if hasattr(lit, 'negated') and not lit.negated and lit.predicate in ['agg_sum', 'agg_avg', 'agg_count', 'agg_max', 'agg_min']:
                    has_aggregates = True
                    # Last term is the result variable
                    if lit.terms and hasattr(lit.terms[-1], 'name'):
                        agg_result_vars.append(lit.terms[-1].name)
            
            logger.debug(f"[QUERY_STRING] Has aggregates: {has_aggregates}, Agg result vars: {agg_result_vars}")

            # Check for negated literals in the query
            neg_literals = [lit for lit in body_literals if hasattr(lit, 'negated') and lit.negated]
            
            # Check if this is a ground query (no variables)
            is_ground_query = True
            for lit in body_literals:
                for term in lit.terms:
                    if isinstance(term, Variable):
                        is_ground_query = False
                        break
                if not is_ground_query:
                    break
                    
            # Determine which variables should be in the query head
            if is_ground_query:
                # For ground queries like emp_name('e_bob', 'bob').
                # We'll add a special variable indicating if the query succeeded
                head_vars = [Variable("proved")]
                logger.debug(f"[QUERY_STRING] Detected ground query, using special 'proved' variable")
            elif has_aggregates:
                # For an aggregate query, extract variables from the aggregate predicates
                # and include any grouping variables as head variables for projection
                query_vars = set()
                for lit in body_literals:
                    if lit.predicate in ['agg_sum', 'agg_avg', 'agg_count', 'agg_max', 'agg_min'] and len(lit.terms) >= 3:
                        # Add the aggregate result variable
                        query_vars.add(lit.terms[-1])
                        # Add any group-by variables
                        for i in range(len(lit.terms) - 2):
                            query_vars.add(lit.terms[i])
                head_vars = list(query_vars)
            elif neg_literals:
                # For negation queries, use all variables from positive literals as head variables
                pos_vars = set()
                for lit in [l for l in body_literals if not (hasattr(l, 'negated') and l.negated)]:
                    for term in lit.terms:
                        if isinstance(term, Variable):
                            pos_vars.add(term)
                head_vars = list(pos_vars) if pos_vars else [Variable("__dummy__")]
            else:
                # For regular queries without aggregation or negation, use all variables from the literals
                # Ensure we use the variables specified by the user
                if variables:
                    head_vars = [Variable(var) for var in variables]
                else:
                    # If no variables were specified, collect all variables from positive literals
                    all_vars = set()
                    for lit in [l for l in body_literals if not (hasattr(l, 'negated') and l.negated)]:
                        for term in lit.terms:
                            if isinstance(term, Variable):
                                all_vars.add(term)
                    head_vars = list(all_vars) if all_vars else [Variable("__dummy__")]
                
            # Create a synthetic head literal with the appropriate variables
            head = Literal("__query__", head_vars)
            
            # Create a rule with the head and body literals, and add comparisons separately
            query_rule = Rule(head=head, body_literals=body_literals, comparisons=comparisons)
            logger.debug(f"[QUERY_STRING] Created query rule: {query_rule}")

            # Create an evaluator and evaluate the query
            # Import here to avoid circular import
            from .evaluator import BottomUpEvaluator
            
            # Temporarily redirect warnings for this block if it's an aggregate query
            if has_aggregates:
                old_log_level = logger.level
                logger.setLevel(logging.ERROR)  # Suppress warnings temporarily
                
            evaluator = BottomUpEvaluator(self)  # Pass the database, not just relations
            evaluator._evaluate_rule(query_rule)  # Use _evaluate_rule method with underscore
            
            # Restore log level if needed
            if has_aggregates:
                logger.setLevel(old_log_level)
                
            logger.debug(f"[QUERY_STRING] Evaluated query rule")

            # Get the result from the evaluator
            if "__query__" in evaluator._full:
                result = evaluator._full["__query__"]
                logger.debug(f"[QUERY_STRING] Found result: {result.num_rows()} rows")

                # Handle empty result case
                if result.num_rows() == 0:
                    logger.debug(f"[QUERY_STRING] Empty result - returning QueryResult.empty")
                    return QueryResult.empty()

                # Remove placeholder column if it exists
                if "__dummy__" in result.columns():
                    logger.debug(f"[QUERY_STRING] Removing dummy column")
                    # Skip return here

                # For aggregate queries, we need special handling
                if has_aggregates:
                    # Check for NaN values in the result using the Frame implementation
                    nan_columns = []
                    all_columns = result.columns()
                    for col in all_columns:
                        try:
                            # Use the Frame implementation to check for NaN-only columns
                            if result.has_only_nan_values(col):
                                nan_columns.append(col)
                        except Exception as e:
                            logger.debug(f"[QUERY_STRING] Error checking NaN in column {col}: {e}")
                            continue

                    # Filter out columns that contain only NaN values
                    if nan_columns:
                        logger.debug(f"[QUERY_STRING] Found NaN columns: {nan_columns}, filtering them out")
                        good_columns = [col for col in all_columns if col not in nan_columns]
                        result = result.project(good_columns)
                        logger.debug(f"[QUERY_STRING] After filtering, columns are: {result.columns()}")
                    
                    # Make sure the result includes the aggregate result variables
                    for var in agg_result_vars:
                        if var not in result.columns():
                            logger.warning(f"[QUERY_STRING] Aggregate result variable {var} not found in result columns")

                logger.debug(f"[QUERY_STRING] Returning QueryResult with columns: {result.columns()}")
                
                # Clean up __dummy__ column if it exists and isn't the only column
                columns = result.columns()
                if '__dummy__' in columns and len(columns) > 1:
                    # Remove the __dummy__ column as it's just an implementation detail
                    columns_to_keep = [col for col in columns if col != '__dummy__']
                    result = result[columns_to_keep]
                    logger.debug(f"[QUERY_STRING] Removed __dummy__ column, final columns: {result.columns()}")
                    
                return QueryResult(result)
            else:
                logger.debug(f"[QUERY_STRING] No result - returning QueryResult.empty")
                return QueryResult.empty()
        except Exception as e:
            logger.error(f"[QUERY_STRING] Error executing query: {e}", exc_info=True)
            return QueryResult.error(str(e))

    def _extract_variables(self, query_body):
        """Extract all variables from a list of literals in the query body.
        
        This method is designed to work with parsed dict AST nodes, handling nested structures.
        
        Args:
            query_body: A list of parsed AST nodes (dicts) representing the query body
            
        Returns:
            A list of Variable objects found in the query body
        """
        from .model import Variable
        variables = []
        for body_item in query_body:
            if isinstance(body_item, dict):
                extracted = self._extract_variables_from_dict(body_item)
                variables.extend([Variable(var_name) for var_name in extracted])
        return variables



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
        empty_frame = self._make_frame.empty(list(colnames))
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
        candidate = self._make_frame.from_dicts([row], list(schema.colnames))
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
            return self._make_frame.empty([])

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
        return self._make_frame.from_dicts(projected, [v for _, v in var_positions])

    def evaluate(self, max_iterations: int = 1000) -> None:
        """
        Evaluate all rules in the database using a bottom-up approach.
        This method executes the rules to derive all possible facts.
        
        Args:
            max_iterations: Maximum number of iterations for the evaluation
        """
        # Import here to avoid circular import
        from .evaluator import BottomUpEvaluator
        
        # Create evaluator and run the evaluation
        evaluator = BottomUpEvaluator(self, max_iterations)
        evaluator.evaluate(max_iterations)
        evaluator.commit_to_db()

    def reset(self) -> None:
        """
        Clear all stored facts (empty all Frames) and drop all rules.
        Schemas remain declared.
        """
        for pred, (schema, _) in self._relations.items():
            empty_frame = self._make_frame.empty(list(schema.colnames))
            self._relations[pred] = (schema, empty_frame)
        self.rules.clear()
