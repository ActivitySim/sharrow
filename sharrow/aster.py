import ast
import io
import logging
import sys
import tokenize

try:
    from ast import unparse
except ImportError:
    from astunparse import unparse as _unparse

    unparse = lambda *args: _unparse(*args).strip("\n")

logger = logging.getLogger("sharrow.aster")

if sys.version_info >= (3, 8):
    ast_Constant_Type = ast.Constant
    ast_String_value = lambda x: x
else:
    ast_Constant_Type = (ast.Index, ast.Constant)
    ast_String_value = lambda x: x.s if isinstance(x, ast.Str) else x


def _isNone(c):
    if c is None:
        return True
    if isinstance(c, ast_Constant_Type) and c.value is None:
        return True
    return False


def extract_all_name_tokens(command):
    all_names = set()
    if isinstance(command, str):
        command = command.encode()
    for token in tokenize.tokenize(io.BytesIO(command).readline):
        if token.type == tokenize.NAME:
            all_names.add(token.string)
    return all_names


def extract_names(command):
    names = set()
    z = ast.parse(command.lstrip())
    for node in ast.walk(z):
        if isinstance(node, ast.Name):
            names.add(node.id)
    return names


def extract_names_2(command):

    if not isinstance(command, str):
        return set(), dict(), dict()

    z = ast.parse(command.lstrip())

    # Attribute-Name pairs first
    attribute_name_pairs = dict()
    skip_names = set()
    for i in ast.walk(z):
        if isinstance(i, ast.Attribute):
            if not isinstance(i.value, ast.Name):
                # print("skip", i.attr, i.value)
                continue
            k = i.value.id
            a = i.attr
            skip_names.add(id(i.value))
            if k not in attribute_name_pairs:
                attribute_name_pairs[k] = set()
            attribute_name_pairs[k].add(a)

    # Subscript pairs next
    subscript_name_pairs = dict()
    for i in ast.walk(z):
        if isinstance(i, ast.Subscript):
            if not isinstance(i.value, ast.Name):
                # print("skip", i.attr, i.value)
                continue
            k = i.value.id
            try:
                a = i.slice.value.value  # Python <= 3.8
            except AttributeError:
                try:
                    a = (
                        i.slice.value
                    )  # Python == 3.9  # FIXME when dropping support for 3.8
                except AttributeError:
                    a = None
            skip_names.add(id(i.value))
            if a is not None:
                if k not in subscript_name_pairs:
                    subscript_name_pairs[k] = set()
                subscript_name_pairs[k].add(a)

    # Plain names last
    plain_names = set()
    for i in ast.walk(z):
        if isinstance(i, ast.Name):
            if id(i) in skip_names:
                continue
            plain_names.add(i.id)

    return plain_names, attribute_name_pairs, subscript_name_pairs


class RewriteExpression(ast.NodeTransformer):
    def visit_Call(self, node):
        try:
            tag = node.func.id
        except AttributeError:
            tag = None
        if tag == "clip":
            if len(node.args) == 3:
                basic, lower, upper = node.args
            elif len(node.args) == 2:
                basic, lower = node.args
                upper = None
            else:
                raise ValueError("incorrect number of args for clip")
            if _isNone(upper):
                return ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Gt()],
                            comparators=[lower],
                        ),
                        basic,
                        lower,
                    ],
                    keywords=[],
                )
            elif _isNone(lower):
                return ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Lt()],
                            comparators=[upper],
                        ),
                        basic,
                        upper,
                    ],
                    keywords=[],
                )
            else:
                return ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Gt()],
                            comparators=[lower],
                        ),
                        ast.Call(
                            func=ast.Name(id="where", ctx=ast.Load()),
                            args=[
                                ast.Compare(
                                    left=basic,
                                    ops=[ast.Lt()],
                                    comparators=[upper],
                                ),
                                basic,
                                upper,
                            ],
                            keywords=[],
                        ),
                        lower,
                    ],
                    keywords=[],
                )
        elif tag == "piece":
            if len(node.args) == 3:
                basic, lower, upper = node.args
            elif len(node.args) == 2:
                basic, lower = node.args
                upper = None
            else:
                raise ValueError("incorrect number of args for piece")
            if _isNone(upper):
                clip = ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Gt()],
                            comparators=[lower],
                        ),
                        basic,
                        lower,
                    ],
                    keywords=[],
                )
                return ast.BinOp(
                    left=clip,
                    op=ast.Sub(),
                    right=lower,
                )
            elif _isNone(lower):
                return ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Lt()],
                            comparators=[upper],
                        ),
                        basic,
                        upper,
                    ],
                    keywords=[],
                )
            else:
                clip = ast.Call(
                    func=ast.Name(id="where", ctx=ast.Load()),
                    args=[
                        ast.Compare(
                            left=basic,
                            ops=[ast.Gt()],
                            comparators=[lower],
                        ),
                        ast.Call(
                            func=ast.Name(id="where", ctx=ast.Load()),
                            args=[
                                ast.Compare(
                                    left=basic,
                                    ops=[ast.Lt()],
                                    comparators=[upper],
                                ),
                                basic,
                                upper,
                            ],
                            keywords=[],
                        ),
                        lower,
                    ],
                    keywords=[],
                )
                return ast.BinOp(
                    left=clip,
                    op=ast.Sub(),
                    right=lower,
                )
        elif tag == "max":
            if len(node.args) == 2:
                left, right = node.args
            else:
                raise ValueError(
                    "incorrect number of args for max (currently only 2 is allowed)"
                )
            return ast.Call(
                func=ast.Name(id="where", ctx=ast.Load()),
                args=[
                    ast.Compare(
                        left=left,
                        ops=[ast.Lt()],
                        comparators=[right],
                    ),
                    right,
                    left,
                ],
                keywords=[],
            )
        elif tag == "min":
            if len(node.args) == 2:
                left, right = node.args
            else:
                raise ValueError(
                    "incorrect number of args for max (currently only 2 is allowed)"
                )
            return ast.Call(
                func=ast.Name(id="where", ctx=ast.Load()),
                args=[
                    ast.Compare(
                        left=left,
                        ops=[ast.Lt()],
                        comparators=[right],
                    ),
                    left,
                    right,
                ],
                keywords=[],
            )
        else:
            return node


def bool_wrap(subnode):
    if (
        isinstance(subnode, ast.Call)
        and isinstance(subnode.func, ast.Attribute)
        and subnode.func.attr == "bool_"
        and isinstance(subnode.func.value, ast.Name)
        and subnode.func.value.id == "np"
    ):
        return subnode
    return ast.Call(
        func=ast.Attribute(
            ast.Name("np"),
            "bool_",
        ),
        args=[subnode],
        keywords={},
    )


class RewriteForNumba(ast.NodeTransformer):
    def __init__(
        self,
        spacename,
        dim_slots,
        spacevars=None,
        rawname="_inputs",
        rawalias="____",
        digital_encodings=None,
        preferred_spacename=None,
    ):
        self.spacename = spacename
        self.dim_slots = dim_slots
        self.spacevars = spacevars
        self.rawname = rawname
        self.rawalias = rawalias
        self.digital_encodings = digital_encodings or {}
        self.preferred_spacename = preferred_spacename

    def log_event(self, tag, node1=None, node2=None):
        if logger.getEffectiveLevel() <= -1:
            if node1 is None:
                logger.log(
                    5, f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag}"
                )
            elif node2 is None:
                try:
                    unparsed = unparse(node1)
                except:  # noqa: E722
                    unparsed = f"{type(node1)} not unparseable"
                logger.log(
                    5,
                    f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag} [{type(node1).__name__}]= {unparsed}",
                )
            else:
                try:
                    unparsed1 = unparse(node1)
                except:  # noqa: E722
                    unparsed1 = f"{type(node1).__name__} not unparseable"
                try:
                    unparsed2 = unparse(node2)
                except:  # noqa: E722
                    unparsed2 = f"{type(node2).__name__} not unparseable"
                logger.log(
                    5,
                    f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag} [{type(node1).__name__},{type(node2).__name__}]= {unparsed1} => {unparsed2}",
                )

    def generic_visit(self, node):
        self.log_event("generic_visit", node)
        return super().generic_visit(node)

    def _replacement(
        self, attr, ctx, original_node, topname=None, transpose_lead=False
    ):
        if topname is None:
            topname = self.spacename
        pref_topname = self.preferred_spacename or topname

        if self.spacevars is not None:
            if attr not in self.spacevars:
                raise KeyError(f"{topname}..{attr}")
                # return original_node

        dim_slots = self.dim_slots
        if isinstance(self.spacevars, dict):
            dim_slots = self.spacevars[attr]

        if transpose_lead:
            decorated_name = ast.Call(
                func=ast.Name("transpose_leading", ctx=ast.Load()),
                args=[ast.Name(id=f"__{pref_topname}__{attr}", ctx=ast.Load())],
                keywords=[],
            )
        else:
            decorated_name = ast.Name(id=f"__{pref_topname}__{attr}", ctx=ast.Load())

        if isinstance(dim_slots, (tuple, list)):
            if len(dim_slots):
                s = ast.Tuple(
                    elts=[
                        (
                            ast.Name(id=f"_arg{n:02}", ctx=ast.Load())
                            if isinstance(n, int)
                            else n
                        )
                        for n in dim_slots
                    ],
                    ctx=ast.Load(),
                )
                result = ast.Subscript(
                    value=decorated_name,
                    slice=s,
                    ctx=ctx,
                )
            else:
                # no indexing, just name replacement
                result = decorated_name
        else:
            if isinstance(dim_slots, int):
                s = ast.Name(id=f"_arg{dim_slots:02}", ctx=ast.Load())
            else:
                s = dim_slots
            result = ast.Subscript(
                value=decorated_name,
                slice=s,
                ctx=ctx,
            )
        digital_encoding = self.digital_encodings.get(attr, None)
        if digital_encoding is not None:

            dictionary = digital_encoding.get("dictionary", None)
            if dictionary is not None:
                result = ast.Subscript(
                    value=ast.Name(
                        id=f"__encoding_dict__{pref_topname}__{attr}", ctx=ast.Load()
                    ),
                    slice=result,
                    ctx=ctx,
                )
            else:
                missing_value = digital_encoding.get("missing_value", None)
                if missing_value is not None:
                    scale = digital_encoding.get("scale", 1)
                    offset = digital_encoding.get("offset", 0)
                    result = ast.Call(
                        func=ast.Name("digital_decode", cts=ast.Load()),
                        args=[
                            result,
                            ast.Num(scale),
                            ast.Num(offset),
                            ast.Num(missing_value),
                        ],
                        keywords=[],
                    )
                else:
                    scale = digital_encoding.get("scale", 1)
                    offset = digital_encoding.get("offset", 0)
                    if scale != 1:
                        result = ast.BinOp(
                            left=result,
                            op=ast.Mult(),
                            right=ast.Num(scale),
                        )
                    if offset:
                        result = ast.BinOp(
                            left=result,
                            op=ast.Add(),
                            right=ast.Num(offset),
                        )
        self.log_event(f"_replacement({attr}, {topname})", original_node, result)
        return result

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            if (
                node.value.id == self.spacename
                and isinstance(node.slice, ast_Constant_Type)
                and isinstance(ast_String_value(node.slice.value), str)
            ):
                self.log_event(f"visit_Subscript(Constant {node.slice.value})")
                return self._replacement(
                    ast_String_value(node.slice.value), node.ctx, node
                )
            if (
                node.value.id == self.rawalias
                and isinstance(node.slice, ast_Constant_Type)
                and isinstance(ast_String_value(node.slice.value), str)
                and ast_String_value(node.slice.value) in self.spacevars
            ):
                result = ast.Subscript(
                    value=ast.Name(id=self.rawname, ctx=ast.Load()),
                    slice=ast.Constant(
                        self.spacevars[ast_String_value(node.slice.value)]
                    ),
                    ctx=node.ctx,
                )
                self.log_event(
                    f"visit_Subscript(Raw {ast_String_value(node.slice.value)})",
                    node,
                    result,
                )
                return result
        self.log_event("visit_Subscript(no change)", node)
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            if node.value.id == self.spacename:
                return self._replacement(node.attr, node.ctx, node)
            if node.value.id == self.rawalias and node.attr in self.spacevars:
                result = ast.Subscript(
                    value=ast.Name(id=self.rawname, ctx=ast.Load()),
                    slice=ast.Constant(self.spacevars[node.attr]),
                    ctx=node.ctx,
                )
                self.log_event(f"visit_Attribute(Raw {node.attr})", node, result)
                return result
            return node
        else:
            result = ast.Attribute(
                value=self.visit(node.value),
                attr=node.attr,
                ctx=node.ctx,
            )
            self.log_event("visit_Attribute(no change)", node, result)
            return result

    def visit_Name(self, node):
        attr = node.id
        if attr not in self.spacevars:
            self.log_event("visit_Name(no change)", node)
            return node
        if self.spacename == "":
            result = ast.Subscript(
                value=ast.Name(id=self.rawname, ctx=ast.Load()),
                slice=ast.Constant(self.spacevars[attr]),
                ctx=node.ctx,
            )
            self.log_event(f"visit_Name(Constant {attr})", node, result)
            return result
        else:
            result = self._replacement(attr, node.ctx, node, self.spacename)
            self.log_event(f"visit_Name(Replacement {attr})", node, result)
            return result

    def visit_UnaryOp(self, node):
        # convert bitflip `~x` operator into `~np.bool_(x)`
        if isinstance(node.op, ast.Invert):
            return ast.UnaryOp(
                op=node.op,
                operand=bool_wrap(self.visit(node.operand)),
            )
        else:
            return ast.UnaryOp(op=node.op, operand=self.visit(node.operand))

    def visit_BinOp(self, node):
        # convert bitwise binops:
        # `x & y` -> `np.bool_(x) & np.bool_(y)`
        # `x | y` -> `np.bool_(x) | np.bool_(y)`
        # `x ^ y` -> `np.bool_(x) ^ np.bool_(y)`
        left = self.visit(node.left)
        right = self.visit(node.right)

        if isinstance(node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)):

            result = ast.BinOp(
                left=bool_wrap(left),
                op=node.op,
                right=bool_wrap(right),
            )
            self.log_event("visit_BinOp(Replacement)", node, result)
        else:
            result = ast.BinOp(
                left=left,
                op=node.op,
                right=right,
            )
            self.log_event("visit_BinOp(no change)", node, result)
        return result

    def visit_Call(self, node):

        result = None
        # implement ActivitySim's "reverse" skims
        if isinstance(node.func, ast.Attribute) and node.func.attr == "reverse":
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == self.spacename:
                    if len(node.args) == 1 and isinstance(
                        node.args[0], ast_Constant_Type
                    ):
                        result = self._replacement(
                            ast_String_value(node.args[0].value),
                            node.func.ctx,
                            None,
                            transpose_lead=True,
                        )
        # handle clip as a method
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clip":
            if len(node.args) == 1 and len(node.keywords) == 0:
                # single positional arg becomes max
                result = ast.Call(
                    func=ast.Name("max", cts=ast.Load()),
                    args=[self.visit(node.func.value), self.visit(node.args[0])],
                    keywords=[],
                )
            elif len(node.args) == 0 and len(node.keywords) == 1:
                # single keyword argument, what is it?
                if node.keywords[0].arg in ("upper", "max"):
                    # becomes min
                    result = ast.Call(
                        func=ast.Name("min", cts=ast.Load()),
                        args=[
                            self.visit(node.func.value),
                            self.visit(node.keywords[0].value),
                        ],
                        keywords=[],
                    )
                elif node.keywords[0].arg in ("lower", "min"):
                    # becomes max
                    result = ast.Call(
                        func=ast.Name("max", cts=ast.Load()),
                        args=[
                            self.visit(node.func.value),
                            self.visit(node.keywords[0].value),
                        ],
                        keywords=[],
                    )
            elif len(node.args) == 2 and len(node.keywords) == 0:
                # two positional arg becomes max and min
                nested_call = ast.Call(
                    func=ast.Name("min", cts=ast.Load()),
                    args=[self.visit(node.func.value), self.visit(node.args[1])],
                    keywords=[],
                )
                result = ast.Call(
                    func=ast.Name("max", cts=ast.Load()),
                    args=[nested_call, self.visit(node.args[0])],
                    keywords=[],
                )
            else:
                # two-way clip with keywords
                # move clip when used as a class method to be a regular function
                # we provide a basic implementation of clip in sharrow.maths
                clip_args = [self.visit(node.func.value)]
                for a in node.args:
                    clip_args.append(self.visit(a))
                result = ast.Call(
                    func=ast.Name("clip", cts=ast.Load()),
                    args=clip_args,
                    keywords=[self.visit(i) for i in node.keywords],
                )
        # move x.apply(func, **kwargs) to be func(x, **kwargs)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "apply":
            apply_args = [self.visit(node.func.value)]
            assert len(node.args) == 1
            apply_func = self.visit(node.args[0])
            result = ast.Call(
                func=apply_func,
                args=apply_args,
                keywords=[self.visit(i) for i in node.keywords],
            )
        # implement ActivitySim's "max" skims
        if isinstance(node.func, ast.Attribute) and node.func.attr == "max":
            if isinstance(node.func.value, ast.Name):
                if node.func.value.id == self.spacename:
                    if len(node.args) == 1 and isinstance(
                        node.args[0], ast_Constant_Type
                    ):
                        forward = self._replacement(
                            ast_String_value(node.args[0].value), node.func.ctx, None
                        )
                        backward = self._replacement(
                            ast_String_value(node.args[0].value),
                            node.func.ctx,
                            None,
                            transpose_lead=True,
                        )
                        result = ast.Call(
                            func=ast.Name("max", ctx=ast.Load()),
                            # func=ast.Attribute(
                            #     value=ast.Name("np", ctx=ast.Load()),
                            #     attr="maximum",
                            # ),
                            args=[forward, backward],
                            keywords=[],
                        )
        # change `x.astype(int, **kwargs)` to `int(x, **kwargs)`
        if isinstance(node.func, ast.Attribute) and node.func.attr == "astype":
            apply_args = [self.visit(node.func.value)]
            assert len(node.args) == 1
            apply_func = self.visit(node.args[0])
            result = ast.Call(
                func=apply_func,
                args=apply_args,
                keywords=[self.visit(i) for i in node.keywords],
            )

        # if no other changes
        if result is None:
            args = [self.visit(i) for i in node.args]
            kwds = [self.visit(i) for i in node.keywords]
            result = ast.Call(
                func=self.visit(node.func),
                args=args,
                keywords=kwds,
            )
        self.log_event("visit_Call", node, result)
        return result


def expression_for_numba(
    expr,
    spacename,
    dim_slots,
    spacevars=None,
    rawname="_inputs",
    rawalias="____",
    digital_encodings=None,
    prefer_name=None,
):
    return unparse(
        RewriteForNumba(
            spacename,
            dim_slots,
            spacevars,
            rawname,
            rawalias,
            digital_encodings,
            prefer_name,
        ).visit(ast.parse(expr))
    )


class Asterize:
    def __init__(self):
        self._cache = {}

    def __call__(self, expr):
        target, result = self._cache.get(expr, (None, None))
        if result is None:
            tree = ast.parse(expr, mode="exec")
            new_tree = ast.fix_missing_locations(RewriteExpression().visit(tree))
            if isinstance(new_tree.body[0], ast.Assign):
                a = new_tree.body[0]
                if not len(a.targets) == 1 or not isinstance(a.targets[0], ast.Name):
                    raise ValueError(
                        f"only one simple named assignment target can be given: {expr}"
                    )
                target = a.targets[0].id
                new_tree = a.value
            result = unparse(new_tree)
            self._cache[expr] = (target, result)
        return target, result


def SimpleSplitter(x):
    if "=" in x:
        s = x.split("=")
        if len(s) == 2:
            return s[0].strip(), s[1].strip()
        else:
            raise ValueError(
                f"only one simple named assignment target can be given: {x}"
            )
    else:
        return None, x
