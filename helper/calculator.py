"""A scientific calculator for the models: calculate("sqrt(4761)") -> "69".

The expression is parsed as Python and its syntax tree walked here: numbers, + - * / // % **
(^ is a power too, as on a calculator; 5! a factorial), brackets, and the functions and constants
below. Nothing else is allowed (no other names, attributes, strings or imports), so it can never
touch files or the network.

Every number is a double-precision float, as on a scientific calculator: about 15 significant
digits, whole numbers exact up to 2^53. A result larger than about 1.8e308 is an overflow, and a
nonzero one smaller than about 2.2e-308 (or one that comes out 0 only because it is that small) is
an underflow; both are errors rather than inf or 0.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import sys

MAX_CHARS = 1000
MAX_NODES = 300
MAX_SHOWN = 15             # significant digits of a result
WHOLE_SHOWN = 1e15         # whole results below this are shown in full, without an exponent
LARGEST = sys.float_info.max
SMALLEST = sys.float_info.min
OVERFLOW = "overflow: the result is larger than about 1.8e308"
UNDERFLOW = "underflow: the result is nonzero but smaller than about 2.2e-308"


class CalcError(ValueError):
    """What went wrong, in words the model can act on."""


def _whole(name, value) -> int:
    if not float(value).is_integer():
        raise CalcError(f"{name} needs whole numbers")
    return int(value)


def _cbrt(x):
    return math.cbrt(x) if hasattr(math, "cbrt") else math.copysign(abs(x) ** (1 / 3), x)


def _log(x, base=None):
    return math.log(x) if base is None else math.log(x, base)


def _root(x, n):
    if n == 0:
        raise CalcError("the 0th root is not defined")
    if x < 0 and float(n).is_integer() and int(n) % 2:
        return -((-x) ** (1 / n))
    return _power(x, 1 / n)


def _exp(x):
    if x < -745.2:
        raise CalcError(UNDERFLOW)
    return math.exp(x)


def _factorial(n):
    n = _whole("factorial", n)
    if n < 0:
        raise CalcError("factorial is not defined for negative numbers; use gamma(x + 1) for others")
    if n > 170:
        raise CalcError(OVERFLOW)
    return float(math.factorial(n))


def _choose(name, n, k, ordered):
    n, k = _whole(name, n), _whole(name, k)
    if n < 0 or k < 0:
        raise CalcError(f"{name} needs numbers of 0 or more")
    if k > n:
        return 0.0
    # Its size first, from log-gamma, so a huge one is an overflow before it is worked out.
    size = math.lgamma(n + 1) - math.lgamma(n - k + 1) - (0 if ordered else math.lgamma(k + 1))
    if size > 709.8:
        raise CalcError(OVERFLOW)
    return float(math.perm(n, k) if ordered else math.comb(n, k))


def _degrees_trig(function):
    return lambda x: function(math.radians(x))


def _round(x, digits=0):
    return float(round(x, _whole("round", digits)))


FUNCTIONS = {
    "sqrt": math.sqrt, "cbrt": _cbrt, "root": _root, "exp": _exp,
    "log": _log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "arcsin": math.asin, "arccos": math.acos, "arctan": math.atan,
    "sind": _degrees_trig(math.sin), "cosd": _degrees_trig(math.cos), "tand": _degrees_trig(math.tan),
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
    "factorial": _factorial, "gamma": math.gamma,
    "comb": lambda n, k: _choose("comb", n, k, False), "perm": lambda n, k: _choose("perm", n, k, True),
    "gcd": lambda *v: float(math.gcd(*[_whole("gcd", x) for x in v])),
    "lcm": lambda *v: float(math.lcm(*[_whole("lcm", x) for x in v])),
    "floor": lambda x: float(math.floor(x)), "ceil": lambda x: float(math.ceil(x)),
    "abs": abs, "round": _round, "min": min, "max": max,
}
CONSTANTS = {"pi": math.pi, "π": math.pi, "e": math.e, "tau": math.tau}

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}

# How models and people write math that Python does not read.
_WRITTEN = (("×", "*"), ("·", "*"), ("÷", "/"), ("−", "-"), ("√", "sqrt"), ("²", "**2"), ("³", "**3"),
            ("^", "**"))
_FACTORIAL = re.compile(r"(\d+(?:\.\d+)?|\))\s*!(?!=)")


def _rewrite(expression: str) -> str:
    expression = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", expression)
    for written, python in _WRITTEN:
        expression = expression.replace(written, python)
    # "5!" and "(2+3)!": a calculator's factorial.
    while True:
        match = _FACTORIAL.search(expression)
        if not match:
            return expression
        end = match.end(1)
        if match.group(1) == ")":
            depth, start = 0, end - 1
            while start >= 0:
                depth += {")": 1, "(": -1}.get(expression[start], 0)
                if depth == 0:
                    break
                start -= 1
            if start < 0:
                raise CalcError("a bracket is not closed")
        else:
            start = match.start(1)
        expression = expression[:start] + "factorial(" + expression[start:end] + ")" + expression[match.end():]


def _checked(value, *inputs) -> float:
    """A float in range: never inf, never NaN, never too small to hold."""
    if isinstance(value, complex):
        raise CalcError("the result is not a real number")
    try:
        value = float(value)
    except OverflowError:
        raise CalcError(OVERFLOW) from None
    if math.isnan(value):
        raise CalcError("the result is not a number")
    if math.isinf(value) or abs(value) > LARGEST:
        raise CalcError(OVERFLOW)
    if 0 < abs(value) < SMALLEST or (value == 0 and inputs and all(x != 0 for x in inputs)):
        raise CalcError(UNDERFLOW)
    return value


def _power(base, exponent):
    if base < 0 and not float(exponent).is_integer():
        raise CalcError("a negative number to a fractional power is not a real number")
    if base == 0 and exponent < 0:
        raise ZeroDivisionError
    # 0.5 ** 2000 and 2 ** -2000 come out 0 only because they are too small to hold.
    return _checked(base ** exponent, base)


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("only numbers are allowed")
        return _checked(node.value)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if node.id in FUNCTIONS:
            raise CalcError(f"{node.id} is a function: write {node.id}(x)")
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Pow):
        return _power(_evaluate(node.left), _evaluate(node.right))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        value = BINARY[type(node.op)](left, right)
        # A product or quotient of nonzero numbers that comes out 0 was too small to hold.
        inputs = (left, right) if isinstance(node.op, ast.Mult) else (left,) if isinstance(node.op, ast.Div) else ()
        return _checked(value, *inputs)
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            raise CalcError(f"unknown function '{name}'")
        if node.keywords:
            raise CalcError("functions take numbers only, not name=value")
        return _checked(FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args]))
    if isinstance(node, (ast.Tuple, ast.List)):
        raise CalcError("one expression at a time; a comma only separates a function's numbers")
    raise CalcError(f"'{ast.unparse(node)}' is not something the calculator does")


def show(value: float) -> str:
    """A result the way a calculator shows it: whole numbers below 1e15 in full, others to 15 digits."""
    if value.is_integer() and abs(value) < WHOLE_SHOWN:
        return str(int(value))
    return format(value, f".{MAX_SHOWN}g")


def calculate(expression) -> str:
    """The result as text, or CalcError saying what is wrong."""
    if not isinstance(expression, str) or not expression.strip():
        raise CalcError("give an expression, such as sqrt(4761)")
    if len(expression) > MAX_CHARS:
        raise CalcError(f"an expression is limited to {MAX_CHARS:,} characters")
    try:
        tree = ast.parse(_rewrite(expression.strip()), mode="eval")
    except SyntaxError:
        raise CalcError("the expression could not be read; check the brackets and operators") from None
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError("the expression is too long; split it into steps")
    try:
        return show(_evaluate(tree))
    except CalcError:
        raise
    except ZeroDivisionError:
        raise CalcError("division by zero") from None
    except OverflowError:
        raise CalcError(OVERFLOW) from None
    except (ValueError, TypeError) as error:
        text = str(error)
        raise CalcError("math domain error: that value is outside what the function takes"
                        if "domain" in text else text) from None
