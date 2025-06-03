import logging
logger = logging.getLogger("datalog.parser")
if not logger.hasHandlers():
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logger.addHandler(handler)
logger.setLevel(logging.WARNING)

from lark import Lark, Transformer, v_args

datalog_grammar = r"""
// -----------------------------
// Top-Level: A program is zero or more facts or rules
// -----------------------------
?start: program | query
program: (fact | rule)*
query: body "."

// -----------------------------
// Facts:  atom "." 
// -----------------------------
fact: pred_atom "."

// -----------------------------
// Rules: head ":-" body "."
// -----------------------------
rule: head ":-" body "."

// The head of a rule is always a predicate atom
head: pred_atom

// Body: one or more literals separated by commas
?body: literal ("," literal)*
?literal: comparison
        | pred_atom
        | neg_literal

neg_literal: "not" _WS pred_atom
_WS: /[ \t]+/

// Predicate atom: ID "(" [ term_list ] ")"
pred_atom: ID "(" term_list? ")"

// Term list: one or more terms separated by commas
term_list: term ("," term)*

// A term is either a variable, a number, a string, or a constant identifier
?term: VARIABLE           // e.g. X, Node1
     | NUMBER             // e.g. 123, 42
     | STRING             // e.g. "hello"
     | CONSTANT           // e.g. a, foo, bar123

// Comparison: term comparator term
comparison: term comparator term

// Comparator operators as explicit tokens
LT: "<"
LE: "<="
GT: ">"
GE: ">="
EQ: "="
NE: "!="

// Combine comparator as alias of explicit tokens
comparator: LT | LE | GT | GE | EQ | NE

// Tokens (lexical items)
VARIABLE: /[A-Z][A-Za-z0-9_]*/
CONSTANT: /[a-z][A-Za-z0-9_]*/
NUMBER:    /[0-9]+/
// Accept both double-quoted and single-quoted strings
STRING:    /("([^"\\]|\\.)*")|('([^'\\]|\\.)*')/

// ID for predicate names (allow underscores)
ID: /[A-Za-z_][A-Za-z0-9_]*/

%import common.CPP_COMMENT
COMMENT_ML: /\/\*[\s\S]*?\*\//
%ignore CPP_COMMENT
%ignore COMMENT_ML
%ignore /#[^\n]*/
%import common.WS
%ignore WS
"""

class DatalogTransformer(Transformer):
    def query(self, items):
        logger.debug("Entering query with items: %s", items)
        result = {"type": "query", "body": items[0]}
        logger.debug("query result: %s", result)
        return result
    """
    Transforms a Lark parse tree into a simple AST represented by nested Python dicts.
    """

    def __init__(self):
        super().__init__()

    def program(self, items):
        logger.debug("Entering program with items: %s", items)
        result = {"type": "program", "clauses": items}
        logger.debug("program result: %s", result)
        return result

    def fact(self, items):
        logger.debug("Entering fact with items: %s", items)
        atom = items[0]
        result = {"type": "fact", "atom": atom}
        logger.debug("fact result: %s", result)
        return result

    def rule(self, items):
        logger.debug("Entering rule with items: %s", items)
        head = items[0]
        body_literals = items[1]
        if head['name'] == 'path':
            logger.debug("Rule head is 'path'!")
        result = {"type": "rule", "head": head, "body": body_literals}
        logger.debug("rule result: %s", result)
        return result

    def head(self, items):
        logger.debug("Entering head with items: %s", items)
        result = items[0]
        logger.debug("head result: %s", result)
        return result

    def body(self, items):
        logger.debug("Entering body with items: %s", items)
        result = items
        logger.debug("body result: %s", result)
        return result

    def neg_literal(self, items):
        # items[1] is pred_atom (items[0] is 'not' or whitespace)
        result = {"type": "negation", "atom": items[-1]}
        logger.debug("neg_literal result: %s", result)
        return result

    def pred_atom(self, items):
        logger.debug("Entering pred_atom with items: %s", items)
        name = str(items[0])
        if len(items) == 2:
            terms = items[1]
        else:
            terms = []
        result = {"type": "atom", "name": name, "terms": terms}
        logger.debug("pred_atom result: %s", result)
        return result

    def term_list(self, items):
        logger.debug("Entering term_list with items: %s", items)
        result = items
        logger.debug("term_list result: %s", result)
        return result

    def term(self, items):
        value = items[0]
        # value is a Token with .type and .value
        if hasattr(value, 'type') and value.type == "NUMBER":
            result = {"type": "number", "value": int(value)}
        elif hasattr(value, 'type') and value.type == "STRING":
            s = value[1:-1].encode("utf-8").decode("unicode_escape")
            result = {"type": "string", "value": s}
        elif hasattr(value, 'type') and value.type == "VARIABLE":
            # Properly recognize variables!
            result = {"type": "variable", "name": str(value)}
        elif hasattr(value, 'type') and value.type == "CONSTANT":
            result = {"type": "constant", "value": str(value)}
        else:
            # fallback for any other token
            result = {"type": "constant", "value": str(value)}
        logger.debug("term result: %s", result)
        return result

    def comparison(self, items):
        logger.debug("Entering comparison with items: %s", items)
        left = items[0]
        op = items[1]
        right = items[2]
        logger.debug("comparison components -> left: %s, op: %s, right: %s", left, op, right)
        result = {"type": "comparison", "op": str(op), "left": left, "right": right}
        logger.debug("comparison result: %s", result)
        return result

    def comparator(self, items):
        logger.debug("Entering comparator with items: %s", items)
        result = items[0]
        logger.debug("comparator result: %s", result)
        return result

    def LT(self, tok):
        logger.debug("LT token: %s", tok)
        return tok
    def LE(self, tok):
        logger.debug("LE token: %s", tok)
        return tok
    def GT(self, tok):
        logger.debug("GT token: %s", tok)
        return tok
    def GE(self, tok):
        logger.debug("GE token: %s", tok)
        return tok
    def EQ(self, tok):
        logger.debug("EQ token: %s", tok)
        return tok
    def NE(self, tok):
        logger.debug("NE token: %s", tok)
        return tok

    def ID(self, tok):
        logger.debug("ID token: %s", tok)
        return tok  # Return the token object, not str(tok)

    def VARIABLE(self, tok):
        logger.debug("VARIABLE token: %s", tok)
        return tok  # Return the token object, not str(tok)

    def CONSTANT(self, tok):
        logger.debug("CONSTANT token: %s", tok)
        return tok  # Return the token object, not str(tok)

    def NUMBER(self, tok):
        logger.debug("NUMBER token: %s", tok)
        return tok

    def STRING(self, tok):
        logger.debug("STRING token: %s", tok)
        return tok


class DatalogParser:
    def __init__(self):
        self.parser = Lark(datalog_grammar, parser="earley")
        self.transformer = DatalogTransformer()

    def parse(self, text):
        logger.debug("Starting parse for text:\n%s", text)
        parse_tree = self.parser.parse(text)
        logger.debug("Parse tree:\n%s", parse_tree.pretty())
        ast = self.transformer.transform(parse_tree)
        logger.debug("Final AST:\n%s", ast)
        return ast
