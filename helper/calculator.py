"""A scientific calculator for the models: calculate("sqrt(4761)") -> "69".

The expression is parsed as Python and its syntax tree walked here: numbers, + - * / // % **
(^ is a power too, as on a calculator; 5! a factorial), brackets, and the functions and constants below. Nothing
else is allowed (no other names, attributes, strings or imports), so it can never touch files or
the network. Whole numbers are exact at any size; sizes are capped so every answer is quick.
"""

from __future__ import annotations

import ast
import math
import operator
import re

MAX_CHARS = 1000
MAX_NODES = 300
MAX_DIGITS = 4000          # of any whole number along the way; Python prints up to 4,300
MAX_FACTORIAL = 1400       # 1400! has 3,799 digits
MAX_COMB = 10_000
MAX_SHOWN = 15             # significant digits of a decimal result


class CalcError(ValueError):
    """What went wrong, in words the model can act on."""


def _cbrt(x):
    return math.copysign(abs(x) ** (1 / 3), x) if not hasattr(math, "cbrt") else math.cbrt(x)


def _log(x, base=None):
    return math.log(x) if base is None else math.log(x, base)


def _root(x, n):
    if n == 0:
        raise CalcError("the 0th root is not defined")
    if x < 0 and int(n) == n and int(n) % 2:
        return -((-x) ** (1 / n))
    return x ** (1 / n)


def _factorial(n):
    if isinstance(n, float):
        if not n.is_integer():
            raise CalcError("factorial needs a whole number; use gamma(x + 1) for others")
        n = int(n)
    if n > MAX_FACTORIAL:
        raise CalcError(f"factorial is limited to {MAX_FACTORIAL}!")
    return math.factorial(n)


def _whole(name, *values, most=None):
    out = []
    for value in values:
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or (most and value > most):
            raise CalcError(f"{name} needs whole numbers" + (f" up to {most:,}" if most else ""))
        out.append(value)
    return out


def _degrees_trig(function):
    return lambda x: function(math.radians(x))


def _round(x, digits=0):
    return round(x, int(digits)) if digits else round(x)


FUNCTIONS = {
    "sqrt": math.sqrt, "cbrt": _cbrt, "root": _root, "exp": math.exp,
    "log": _log, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sind": _degrees_trig(math.sin), "cosd": _degrees_trig(math.cos), "tand": _degrees_trig(math.tan),
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "asinh": math.asinh, "acosh": math.acosh, "atanh": math.atanh,
    "degrees": math.degrees, "radians": math.radians, "hypot": math.hypot,
    "factorial": _factorial, "gamma": math.gamma,
    "comb": lambda n, k: math.comb(*_whole("comb", n, k, most=MAX_COMB)),
    "perm": lambda n, k: math.perm(*_whole("perm", n, k, most=MAX_COMB)),
    "gcd": lambda *v: math.gcd(*_whole("gcd", *v)), "lcm": lambda *v: math.lcm(*_whole("lcm", *v)),
    "floor": math.floor, "ceil": math.ceil, "abs": abs, "round": _round, "min": min, "max": max,
}
FUNCTIONS["arcsin"], FUNCTIONS["arccos"], FUNCTIONS["arctan"] = math.asin, math.acos, math.atan
CONSTANTS = {"pi": math.pi, "π": math.pi, "e": math.e, "tau": math.tau}

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
          ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: None}
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


def _digits(value):
    if isinstance(value, int) and not isinstance(value, bool) and value.bit_length() > MAX_DIGITS * 3.33:
        raise CalcError(f"the number is longer than {MAX_DIGITS:,} digits")
    if isinstance(value, complex):
        raise CalcError("the result is not a real number")
    return value


def _power(base, exponent):
    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and abs(base) > 1:
        if exponent * math.log10(abs(base)) > MAX_DIGITS:
            raise CalcError(f"the number is longer than {MAX_DIGITS:,} digits")
    if isinstance(base, (int, float)) and base < 0 and isinstance(exponent, float) and not exponent.is_integer():
        raise CalcError("a negative number to a fractional power is not a real number")
    return base ** exponent


def _evaluate(node):
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise CalcError("only numbers are allowed")
        return _digits(node.value)
    if isinstance(node, ast.Name):
        if node.id in CONSTANTS:
            return CONSTANTS[node.id]
        if node.id in FUNCTIONS:
            raise CalcError(f"{node.id} is a function: write {node.id}(x)")
        raise CalcError(f"unknown name '{node.id}'")
    if isinstance(node, ast.UnaryOp) and type(node.op) in UNARY:
        return UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            return _digits(_power(left, right))
        if isinstance(node.op, ast.Mult) and isinstance(left, int) and isinstance(right, int):
            if (abs(left).bit_length() + abs(right).bit_length()) > MAX_DIGITS * 3.33:
                raise CalcError(f"the number is longer than {MAX_DIGITS:,} digits")
        if isinstance(node.op, ast.Div) and isinstance(left, int) and isinstance(right, int) and right \
                and left % right == 0:
            return left // right
        return _digits(BINARY[type(node.op)](left, right))
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            raise CalcError(f"unknown function '{name}'")
        if node.keywords:
            raise CalcError("functions take numbers only, not name=value")
        return _digits(FUNCTIONS[node.func.id](*[_evaluate(arg) for arg in node.args]))
    if isinstance(node, (ast.Tuple, ast.List)):
        raise CalcError("one expression at a time; a comma only separates a function's numbers")
    raise CalcError(f"'{ast.unparse(node)}' is not something the calculator does")


def show(value) -> str:
    """A result the way a calculator shows it: whole numbers in full, others to 15 digits."""
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            raise CalcError("the result is not a finite number")
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return format(value, f".{MAX_SHOWN}g")
    return str(value)


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
        raise CalcError("the result is too large") from None
    except (ValueError, TypeError) as error:
        text = str(error)
        raise CalcError("math domain error: that value is outside what the function takes"
                        if "domain" in text else text) from None
