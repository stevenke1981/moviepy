"""Finite, capability-free values and a bounded expression AST interpreter."""

import ast
import math
import operator
from types import MappingProxyType

import numpy as np


_NUMERIC_TYPES = frozenset(
    (
        int,
        float,
        np.int8,
        np.int16,
        np.int32,
        np.int64,
        np.uint8,
        np.uint16,
        np.uint32,
        np.uint64,
        np.float16,
        np.float32,
        np.float64,
        np.longdouble,
    )
)


class ExpressionError(ValueError):
    """Reject an invalid, unsafe, or resource-exhausting expression."""


def safe_number(value):
    """Convert an exact inert scalar to a finite bounded float."""
    if type(value) not in _NUMERIC_TYPES:
        raise ExpressionError("expected an inert numeric scalar")
    if type(value) is int and value.bit_length() > 333:
        raise ExpressionError("number exceeds magnitude limit")
    number = float(value)
    if not math.isfinite(number) or abs(number) > 1e100:
        raise ExpressionError("number must be finite and bounded")
    return number


def sanitize(value, depth=0, budget=None):
    """Detach data into bounded primitives without calling user protocols."""
    if budget is None:
        budget = [1024]
    budget[0] -= 1
    if depth > 8 or budget[0] < 0:
        raise ExpressionError("value nesting or total size exceeds limit")
    if type(value) in (bool, type(None)):
        return value
    if type(value) is str:
        if len(value) > 256:
            raise ExpressionError("string exceeds limit")
        return value
    if type(value) in _NUMERIC_TYPES:
        return safe_number(value)
    if type(value) is np.ndarray:
        if value.dtype.kind not in "iuf" or value.ndim != 1 or value.size > 64:
            raise ExpressionError("only short one-dimensional numeric arrays")
        return tuple(sanitize(x, depth + 1, budget) for x in value)
    if type(value) in (tuple, list):
        if len(value) > 64:
            raise ExpressionError("sequence exceeds limit")
        return tuple(sanitize(x, depth + 1, budget) for x in value)
    if type(value) is ExpressionSnapshot:
        return ExpressionSnapshot._from_dict(value._fields, depth, budget)
    if type(value) is dict:
        return ExpressionSnapshot._from_dict(value, depth, budget)
    raise ExpressionError("unsupported value type")


class ExpressionSnapshot:
    """Expose a detached, immutable mapping as safe expression metadata.

    Parameters
    ----------
    data : dict
        Plain mapping of public names to finite primitives or nested mappings.

    Notes
    -----
    Attributes are interpreted as mapping keys; Python attributes and methods
    are never exposed. Numeric sequences contain at most 64 components.
    """

    __slots__ = ("_fields",)

    def __init__(self, data):
        if type(data) is not dict:
            raise ExpressionError("snapshot requires a plain dictionary")
        clean = self._from_dict(data, 0, [1024])
        object.__setattr__(self, "_fields", clean._fields)

    def __setattr__(self, name, value):
        raise ExpressionError("snapshots are immutable")

    @classmethod
    def _from_dict(cls, data, depth, budget):
        if type(data) not in (dict, MappingProxyType):
            raise ExpressionError("snapshot fields must be inert mappings")
        if depth > 8 or len(data) > 64:
            raise ExpressionError("snapshot exceeds size or depth limit")
        fields = {}
        for key, value in data.items():
            if type(key) is not str or not key.isidentifier() or key.startswith("_"):
                raise ExpressionError("snapshot keys must be public identifiers")
            if len(key) > 256:
                raise ExpressionError("snapshot key exceeds limit")
            fields[key] = sanitize(value, depth + 1, budget)
        instance = object.__new__(cls)
        object.__setattr__(instance, "_fields", MappingProxyType(fields))
        return instance


_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_COMPARE = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}
_ALLOWED = (
    (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.List,
        ast.Tuple,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Subscript,
        ast.Attribute,
        ast.Call,
        ast.keyword,
        ast.And,
        ast.Or,
        ast.Not,
        ast.USub,
        ast.UAdd,
    )
    + tuple(_BINARY)
    + tuple(_COMPARE)
)


def parse(source):
    """Validate an expression before executing any arithmetic or helper."""
    if type(source) is not str or not source or len(source) > 8192:
        raise ExpressionError("source must be a nonempty string <=8192 characters")
    try:
        tree = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as error:
        raise ExpressionError("invalid expression syntax") from error
    stack, count = [(tree, 0)], 0
    while stack:
        node, depth = stack.pop()
        count += 1
        if depth > 32 or count > 512 or type(node) not in _ALLOWED:
            raise ExpressionError("unsupported syntax or AST limit exceeded")
        _validate_node(node)
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _validate_node(node):
    if isinstance(node, ast.Name) and node.id.startswith("_"):
        raise ExpressionError("private names are forbidden")
    if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
        raise ExpressionError("private attributes are forbidden")
    if isinstance(node, ast.Constant):
        sanitize(node.value)
    if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) > 64:
        raise ExpressionError("sequence exceeds limit")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ExpressionError("only direct helper calls are allowed")
        names = [keyword.arg for keyword in node.keywords]
        if None in names or len(set(names)) != len(names):
            raise ExpressionError("unpacked or duplicate keywords are forbidden")


def binary(left, right, operation):
    """Apply bounded scalar or componentwise vector arithmetic."""
    if type(left) is tuple or type(right) is tuple:
        if type(left) is not tuple:
            left = (left,) * len(right)
        if type(right) is not tuple:
            right = (right,) * len(left)
        if len(left) != len(right) or any(type(x) is tuple for x in left + right):
            raise ExpressionError("vector shape mismatch")
        return tuple(binary(a, b, operation) for a, b in zip(left, right))
    left, right = safe_number(left), safe_number(right)
    if operation is ast.Pow and abs(right) > 32:
        raise ExpressionError("power exponent exceeds limit")
    return safe_number(_BINARY[operation](left, right))


class Interpreter:
    """Evaluate validated trees with fixed helper capabilities and budgets."""

    def __init__(self, environment, helpers):
        self.environment = environment
        self.helpers = helpers
        self.remaining = 1024

    def evaluate(self, node):
        """Interpret one node while charging the operation budget."""
        self.remaining -= 1
        if self.remaining < 0:
            raise ExpressionError("expression operation budget exceeded")
        method = getattr(self, "_" + type(node).__name__, None)
        if method is None:
            raise ExpressionError("unsupported expression node")
        return method(node)

    def _Expression(self, node):
        return self.evaluate(node.body)

    def _Constant(self, node):
        return sanitize(node.value)

    def _Name(self, node):
        if node.id not in self.environment:
            raise ExpressionError("unknown expression name: " + node.id[:80])
        return self.environment[node.id]

    def _List(self, node):
        return tuple(self.evaluate(item) for item in node.elts)

    _Tuple = _List

    def _BinOp(self, node):
        return binary(
            self.evaluate(node.left), self.evaluate(node.right), type(node.op)
        )

    def _UnaryOp(self, node):
        value = self.evaluate(node.operand)
        if isinstance(node.op, ast.Not):
            return not bool(value)
        return binary(value, -1 if isinstance(node.op, ast.USub) else 1, ast.Mult)

    def _BoolOp(self, node):
        for item in node.values:
            result = self.evaluate(item)
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result

    def _Compare(self, node):
        left = self.evaluate(node.left)
        for operation, comparator in zip(node.ops, node.comparators):
            right = self.evaluate(comparator)
            if type(left) is ExpressionSnapshot or type(right) is ExpressionSnapshot:
                raise ExpressionError("snapshots cannot be compared")
            if not _COMPARE[type(operation)](left, right):
                return False
            left = right
        return True

    def _IfExp(self, node):
        return self.evaluate(node.body if self.evaluate(node.test) else node.orelse)

    def _Attribute(self, node):
        snapshot = self.evaluate(node.value)
        if (
            type(snapshot) is not ExpressionSnapshot
            or node.attr not in snapshot._fields
        ):
            raise ExpressionError("unknown snapshot attribute")
        return snapshot._fields[node.attr]

    def _Subscript(self, node):
        value, key = self.evaluate(node.value), self.evaluate(node.slice)
        if type(value) is ExpressionSnapshot and type(key) is str:
            if key not in value._fields:
                raise ExpressionError("unknown snapshot key")
            return value._fields[key]
        if type(value) is not tuple or type(key) is not float or not key.is_integer():
            raise ExpressionError("indexing requires sequence and integer")
        return value[int(key)]

    def _Call(self, node):
        name = node.func.id
        if name not in self.helpers:
            raise ExpressionError("unknown expression helper: " + name[:80])
        args = [self.evaluate(arg) for arg in node.args]
        kwargs = {item.arg: self.evaluate(item.value) for item in node.keywords}
        return sanitize(self.helpers[name](*args, **kwargs))
