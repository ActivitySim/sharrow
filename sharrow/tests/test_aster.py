import ast

import numpy as np
import pytest

from sharrow.aster import ast_String_value, expression_for_numba
from sharrow.maths import digital_decode, transpose_leading


@pytest.mark.parametrize("value", ["DIST", "", "AM"])
def test_string_constant_value(value):
    assert ast_String_value(ast.Constant(value=value)) == value
    assert ast_String_value(value) == value


@pytest.mark.parametrize(
    "expression", ["1", "1.5", "1j", "True", "None", "b'DIST'", "...", "name"]
)
def test_non_string_node_is_preserved(expression):
    node = ast.parse(expression, mode="eval").body
    assert ast_String_value(node) is node


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("skims['DIST']", 3),
        ("skims.get('DIST')", 3),
        ("skims.get('DIST', 0)", 3),
        ("skims.get('DIST', default=0)", 3),
        ("skims.get('missing', 9)", 9),
        ("skims.reverse('DIST')", 7),
        ("skims.max('DIST')", 7),
    ],
)
def test_string_lookup(expression, expected):
    rewritten = expression_for_numba(
        expression,
        spacename="skims",
        dim_slots=(0, 1),
        spacevars={"DIST"},
        get_default=True,
    )
    namespace = {
        "__skims__DIST": np.array([[1, 3], [7, 9]]),
        "_arg00": 0,
        "_arg01": 1,
        "transpose_leading": transpose_leading.py_func,
    }
    assert eval(rewritten, namespace) == expected


def test_raw_string_lookup():
    rewritten = expression_for_numba(
        "____['income']",
        spacename="",
        dim_slots=(),
        spacevars={"income": 1},
    )
    assert eval(rewritten, {"_inputs": np.array([5, 12])}) == 12


def test_string_tuple_lookup():
    rewritten = expression_for_numba(
        "skims['DIST', 'PM']",
        spacename="skims",
        dim_slots=(0, 1, {"AM": 0, "PM": 1}),
        spacevars={"DIST"},
    )
    namespace = {
        "__skims__DIST": np.arange(8).reshape(2, 2, 2),
        "_arg00": 0,
        "_arg01": 1,
    }
    assert eval(rewritten, namespace) == 3


@pytest.mark.parametrize(
    "encoding, values, expected",
    [
        ({"scale": 2.5}, [0, 4], [0, 10]),
        ({"offset": 5}, [0, 4], [5, 9]),
        ({"scale": 2.5, "offset": 5}, [0, 4], [5, 15]),
        (
            {"scale": 2.5, "offset": 5, "missing_value": -99},
            [-1, 0, 4],
            [-99, 5, 15],
        ),
    ],
)
def test_numeric_decoding(encoding, values, expected):
    rewritten = expression_for_numba(
        "skims.DIST",
        spacename="skims",
        dim_slots=(0,),
        spacevars={"DIST"},
        digital_encodings={"DIST": encoding},
    )
    namespace = {
        "__skims__DIST": np.array(values),
        "digital_decode": digital_decode.py_func,
    }
    result = [eval(rewritten, {**namespace, "_arg00": i}) for i in range(len(values))]
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("code, expected", [(-1, True), (0, True), (1, False)])
def test_categorical_missing_value(code, expected):
    rewritten = expression_for_numba(
        "skims.mode.isna()",
        spacename="skims",
        dim_slots=(0,),
        spacevars={"mode"},
        digital_encodings={"mode": {"dictionary": np.array([np.nan, 1.0])}},
    )
    namespace = {
        "__skims__mode": np.array([code]),
        "__encoding_dict__skims__mode": np.array([np.nan, 1.0]),
        "_arg00": 0,
        "isnan_fast_safe": np.isnan,
    }
    assert bool(eval(rewritten, namespace)) is expected
