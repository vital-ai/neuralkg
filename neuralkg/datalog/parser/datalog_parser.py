from lark import Lark, Transformer, v_args

datalog_grammar = r"""
// -----------------------------
// Top-Level: A program is zero or more facts or rules
// -----------------------------
?start: program
program: (fact | rule)*

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
?literal: comparison      // e.g. V < 200
        | pred_atom       // e.g. edge(a,b)

// Predicate atom: ID "(" [ term_list ] ")"
pred_atom: ID "(" [term_list] ")"

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
STRING:    /"([^"\\]|\\.)*"/

// ID for predicate names (allow underscores)
ID: /[A-Za-z_][A-Za-z0-9_]*/

// Whitespace and comments (ignore all whitespace including newlines)
%import common.WS
%ignore WS
%ignore /#[^\n]*/
"""

class DatalogTransformer(Transformer):
    """
    Transforms a Lark parse tree into a simple AST represented by nested Python dicts.
    Debug print statements added to trace transformation.
    """

    def __init__(self):
        super().__init__()

    def program(self, items):
        print("[DEBUG] Entering program with items:", items)
        result = {"type": "program", "clauses": items}
        print("[DEBUG] program result:", result)
        return result

    def fact(self, items):
        print("[DEBUG] Entering fact with items:", items)
        atom = items[0]
        result = {"type": "fact", "atom": atom}
        print("[DEBUG] fact result:", result)
        return result

    def rule(self, items):
        print("[DEBUG] Entering rule with items:", items)
        head = items[0]
        body_literals = items[1]
        result = {"type": "rule", "head": head, "body": body_literals}
        print("[DEBUG] rule result:", result)
        return result

    def head(self, items):
        print("[DEBUG] Entering head with items:", items)
        result = items[0]
        print("[DEBUG] head result:", result)
        return result

    def body(self, items):
        print("[DEBUG] Entering body with items:", items)
        result = items
        print("[DEBUG] body result:", result)
        return result

    def pred_atom(self, items):
        print("[DEBUG] Entering pred_atom with items:", items)
        name = str(items[0])
        if len(items) == 2:
            terms = items[1]
        else:
            terms = []
        result = {"type": "atom", "name": name, "terms": terms}
        print("[DEBUG] pred_atom result:", result)
        return result

    def term_list(self, items):
        print("[DEBUG] Entering term_list with items:", items)
        result = items
        print("[DEBUG] term_list result:", result)
        return result

    def term(self, items):
        print("[DEBUG] Entering term with items:", items)
        value = items[0]
        if value.type == "NUMBER":
            result = {"type": "number", "value": int(value)}
        elif value.type == "STRING":
            s = value[1:-1].encode("utf-8").decode("unicode_escape")
            result = {"type": "string", "value": s}
        elif value.type == "VARIABLE":
            result = {"type": "variable", "name": str(value)}
        else:
            result = {"type": "constant", "value": str(value)}
        print("[DEBUG] term result:", result)
        return result

    def comparison(self, items):
        print("[DEBUG] Entering comparison with items:", items)
        left = items[0]
        op = items[1]
        right = items[2]
        print(f"[DEBUG] comparison components -> left: {left}, op: {op}, right: {right}")
        result = {"type": "comparison", "op": str(op), "left": left, "right": right}
        print("[DEBUG] comparison result:", result)
        return result

    def comparator(self, items):
        print("[DEBUG] Entering comparator with items:", items)
        result = items[0]
        print("[DEBUG] comparator result:", result)
        return result

    def LT(self, tok):
        print("[DEBUG] LT token:", tok)
        return tok
    def LE(self, tok):
        print("[DEBUG] LE token:", tok)
        return tok
    def GT(self, tok):
        print("[DEBUG] GT token:", tok)
        return tok
    def GE(self, tok):
        print("[DEBUG] GE token:", tok)
        return tok
    def EQ(self, tok):
        print("[DEBUG] EQ token:", tok)
        return tok
    def NE(self, tok):
        print("[DEBUG] NE token:", tok)
        return tok

    def ID(self, tok):
        print("[DEBUG] ID token:", tok)
        return tok

    def VARIABLE(self, tok):
        print("[DEBUG] VARIABLE token:", tok)
        return tok

    def CONSTANT(self, tok):
        print("[DEBUG] CONSTANT token:", tok)
        return tok

    def NUMBER(self, tok):
        print("[DEBUG] NUMBER token:", tok)
        return tok

    def STRING(self, tok):
        print("[DEBUG] STRING token:", tok)
        return tok


class DatalogParser:
    def __init__(self):
        self.parser = Lark(datalog_grammar, parser="earley")
        self.transformer = DatalogTransformer()

    def parse(self, text):
        print("[DEBUG] Starting parse for text:\n", text)
        parse_tree = self.parser.parse(text)
        print("[DEBUG] Parse tree:\n", parse_tree.pretty())
        ast = self.transformer.transform(parse_tree)
        print("[DEBUG] Final AST:\n", ast)
        return ast

