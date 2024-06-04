import ast
import io
import logging
import tokenize
import warnings

import numpy as np

try:
    from ast import unparse
except ImportError:
    from astunparse import unparse as _unparse

    def unparse(*args):
        return _unparse(*args).strip("\n")


logger = logging.getLogger("sharrow.aster")


def unparse_(*args):
    # spit out a ast dump on error
    try:
        return unparse(*args)
    except Exception:
        logger.warning(ast.dump(*args))
        raise


ast_Constant_Type = ast.Constant


def ast_String_value(x):
    return x.value if isinstance(x, ast.Str) else x


ast_TupleIndex_Type = ast.Tuple


def ast_Index_Value(x):
    return x


ast_Constant = ast.Constant


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
        extra_vars=None,
        blenders=None,
        bool_wrapping=False,
        swallow_errors=False,
        get_default=False,
        original_expr="???",
    ):
        self.spacename = spacename
        self.dim_slots = dim_slots
        self.spacevars = spacevars
        self.rawname = rawname
        self.rawalias = rawalias
        self.digital_encodings = digital_encodings or {}
        self.preferred_spacename = preferred_spacename
        self.extra_vars = extra_vars or {}
        self.blenders = blenders or {}
        self.bool_wrapping = bool_wrapping
        self.swallow_errors = swallow_errors
        self.get_default = get_default
        self.original_expr = original_expr

    def log_event(self, tag, node1=None, node2=None):
        if logger.getEffectiveLevel() <= 0:
            if node1 is None:
                logger.debug(f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag}")
            elif node2 is None:
                try:
                    unparsed = unparse_(node1)
                except:  # noqa: E722
                    unparsed = f"{type(node1)} not unparseable"
                logger.debug(
                    f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag} "
                    f"[{type(node1).__name__}]= {unparsed}",
                )
            else:
                try:
                    unparsed1 = unparse_(node1)
                except:  # noqa: E722
                    unparsed1 = f"{type(node1).__name__} not unparseable"
                try:
                    unparsed2 = unparse_(node2)
                except:  # noqa: E722
                    unparsed2 = f"{type(node2).__name__} not unparseable"
                logger.debug(
                    f"RewriteForNumba({self.spacename}|{self.rawalias}).{tag} "
                    f"[{type(node1).__name__},{type(node2).__name__}]= "
                    f"{unparsed1} => {unparsed2}",
                )

    def generic_visit(self, node):
        self.log_event("generic_visit", node)
        return super().generic_visit(node)

    def _replacement(
        self,
        attr,
        ctx,
        original_node,
        topname=None,
        transpose_lead=False,
        missing_dim_value=None,
    ):
        if topname is None:
            topname = self.spacename
        pref_topname = self.preferred_spacename or topname

        if self.spacevars is not None:
            if attr not in self.spacevars:
                if self.get_default or (
                    topname == pref_topname and not self.swallow_errors
                ):
                    raise KeyError(
                        f"{topname}..{attr}\nexpression={self.original_expr}"
                    )
                # we originally raised a KeyError here regardless, but what if
                # we just give back the original node, and see if other spaces,
                # possibly fallback spaces, might work?  If nothing works then
                # it will still eventually error out when compiling?
                # The swallow errors option allows us to continue processing
                # using the same rules, then circle back later to clean up
                # errors, so that as-yet unprocessed Ast elements get the
                # chance to be seen by the first pass.
                return original_node

        dim_slots = self.dim_slots
        if isinstance(self.spacevars, dict):
            dim_slots = self.spacevars[attr]

        def maybe_transpose(thing):
            if transpose_lead:
                return ast.Call(
                    func=ast.Name("transpose_leading", ctx=ast.Load()),
                    args=[thing],
                    keywords=[],
                )
            else:
                return thing

        def _maybe_transpose_first_two_args(_slice):
            if transpose_lead:
                elts = _slice.elts
                if len(elts) >= 2:
                    elts = [elts[1], elts[0], *elts[2:]]
                return type(_slice)(elts=elts)
            else:
                return _slice

        raw_decorated_name = f"__{pref_topname}__{attr}"
        decorated_name = maybe_transpose(
            ast.Name(id=raw_decorated_name, ctx=ast.Load())
        )
        logger.debug(f"    decorated_name= {unparse_(decorated_name)}")

        if isinstance(dim_slots, (tuple, list)):
            if len(dim_slots):
                elts = []
                for n in dim_slots:
                    if isinstance(n, int):
                        elts.append(ast.Name(id=f"_arg{n:02}", ctx=ast.Load()))
                    elif isinstance(n, dict):
                        elts.append(
                            ast.Constant(n=n[missing_dim_value], ctx=ast.Load())
                        )
                    else:
                        elts.append(n)
                    logger.debug(f"ELT {unparse_(elts[-1])}")
                s = ast.Tuple(
                    elts=elts,
                    ctx=ast.Load(),
                )
                result = ast.Subscript(
                    value=decorated_name,
                    slice=s,
                    ctx=ctx,
                )
                logger.debug(f"two+ dim_slots on decorated_name {unparse_(result)}")
            else:
                # no indexing, just name replacement
                result = decorated_name
                logger.debug(f"just decorated_name {unparse_(result)}")
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
            logger.debug(f"one dim_slots on decorated_name {unparse_(result)}")

        digital_encoding = self.digital_encodings.get(attr, None)
        if digital_encoding is not None:
            dictionary = digital_encoding.get("dictionary", None)
            offset_source = digital_encoding.get("offset_source", None)
            if dictionary is not None:
                result = ast.Subscript(
                    value=ast.Name(
                        id=f"__encoding_dict__{pref_topname}__{attr}", ctx=ast.Load()
                    ),
                    slice=result,
                    ctx=ctx,
                )
            elif offset_source is not None:
                result = ast.Subscript(
                    value=maybe_transpose(
                        ast.Name(
                            id=f"__{pref_topname}__{offset_source}", ctx=ast.Load()
                        )
                    ),
                    slice=result.slice,
                    ctx=ctx,
                )
                result = ast.Subscript(
                    value=ast.Name(id=f"__{pref_topname}__{attr}", ctx=ast.Load()),
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

        blender = self.blenders.get(attr, None)
        if blender is not None:
            # get_blended_2(backstop, indices, indptr, data, i, j, blend_limit=np.inf)
            result_args = result.slice.elts
            # inside the blender, the args will be maz-taz mapped, but we need the plain (i)maz too now
            result_arg_ = [j.slice for j in result_args]
            if len(result_args) == 2:
                result = ast.Call(
                    func=ast.Name("get_blended_2", cts=ast.Load()),
                    args=[
                        result,
                        ast.Name(
                            id=f"__{pref_topname}___s_{attr}__indices", ctx=ast.Load()
                        ),
                        ast.Name(
                            id=f"__{pref_topname}___s_{attr}__indptr", ctx=ast.Load()
                        ),
                        ast.Name(
                            id=f"__{pref_topname}___s_{attr}__data", ctx=ast.Load()
                        ),
                        result_arg_[0 if not transpose_lead else 1],
                        result_arg_[1 if not transpose_lead else 0],
                        ast_Constant(blender.get("max_blend_distance")),  # blend_limit
                    ],
                    keywords=[],
                )
            else:
                raise NotImplementedError()

        self.log_event(f"_replacement({attr}, {topname})", original_node, result)
        return result

    def visit_Subscript(self, node):
        if isinstance(node.value, ast.Name):
            # for XXX[YYY], XXX is a space name and YYY is a literal value: skims['DIST']
            if (
                node.value.id == self.spacename
                and isinstance(node.slice, ast_Constant_Type)
                and isinstance(ast_String_value(node.slice), str)
            ):
                self.log_event(f"visit_Subscript(Constant {node.slice.value})")
                return self._replacement(ast_String_value(node.slice), node.ctx, node)
            # for XXX[YYY], XXX is the raw placeholds and YYY is a literal value: ____['income']
            if (
                node.value.id == self.rawalias
                and isinstance(node.slice, ast_Constant_Type)
                and isinstance(ast_String_value(node.slice), str)
                and ast_String_value(node.slice) in self.spacevars
            ):
                result = ast.Subscript(
                    value=ast.Name(id=self.rawname, ctx=ast.Load()),
                    slice=ast.Constant(self.spacevars[ast_String_value(node.slice)]),
                    ctx=node.ctx,
                )
                self.log_event(
                    f"visit_Subscript(Raw {ast_String_value(node.slice)})",
                    node,
                    result,
                )
                return result
            # for XXX[YYY,ZZZ], XXX is a space name and YYY is a literal value and
            # ZZZ is a literal value: skims['SOV_TIME','MD']
            if (
                node.value.id == self.spacename
                and isinstance(ast_Index_Value(node.slice), ast.Tuple)
                and len(ast_Index_Value(node.slice).elts) == 2
                and isinstance(ast_Index_Value(node.slice).elts[0], ast_Constant_Type)
                and isinstance(ast_Index_Value(node.slice).elts[1], ast_Constant_Type)
                and isinstance(
                    ast_String_value(ast_Index_Value(node.slice).elts[0]), str
                )
                and isinstance(
                    ast_String_value(ast_Index_Value(node.slice).elts[1]), str
                )
            ):
                _a = ast_String_value(ast_Index_Value(node.slice).elts[0])
                _b = ast_String_value(ast_Index_Value(node.slice).elts[1])
                self.log_event(f"visit_Subscript(Tuple ({_a}, {_b}))")
                return self._replacement(
                    _a,
                    node.ctx,
                    node,
                    missing_dim_value=_b,
                )
            # for XXX[...], there is no space name and XXX is the name of an aux_var
            if (
                node.value.id in self.spacevars
                and isinstance(self.spacevars[node.value.id], ast.Name)
                and self.spacename == ""
            ):
                result = ast.Subscript(
                    value=self.spacevars[node.value.id],
                    slice=self.visit(node.slice),
                    ctx=node.ctx,
                )
                self.log_event(
                    f"visit_Subscript(AuxVar {node.value.id})",
                    node,
                    result,
                )
                return result
        self.log_event("visit_Subscript(no change)", node)
        return node

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name):
            # for XXX.YYY, XXX is a space name and YYY is a literal value: skims.DIST
            if node.value.id == self.spacename:
                return self._replacement(node.attr, node.ctx, node)
            # for ____.YYY, handles unadorned values in the top level
            if node.value.id == self.rawalias and node.attr in self.spacevars:
                result = ast.Subscript(
                    value=ast.Name(id=self.rawname, ctx=ast.Load()),
                    slice=ast.Constant(self.spacevars[node.attr]),
                    ctx=node.ctx,
                )
                self.log_event(f"visit_Attribute(Raw {node.attr})", node, result)
                return result
            # for YYY.ZZZ, where YYY is a variable in the root and ZZZ is anything
            if self.spacename == "" and node.value.id in self.spacevars:
                result = ast.Attribute(
                    value=self.visit(node.value),
                    attr=node.attr,
                    ctx=node.ctx,
                )
                self.log_event("visit_Attribute(lead change)", node, result)
                return result
            return node
        else:
            # pass through
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
            if isinstance(self.spacevars[attr], ast.Name):
                # when spacevars values are ast.Name we are using it, it's probably an aux_var
                result = self.spacevars[attr]
            else:
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
        if self.bool_wrapping and isinstance(node.op, ast.Invert):
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

        if self.bool_wrapping and isinstance(
            node.op, (ast.BitAnd, ast.BitOr, ast.BitXor)
        ):
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
        if (
            isinstance(node.func, ast.Attribute) and node.func.attr == "reverse"
        ):  # *.reverse(...)
            if isinstance(node.func.value, ast.Name):  # somename.reverse(...)
                if node.func.value.id == self.spacename:  # spacename.reverse(...)
                    if len(node.args) == 1 and isinstance(
                        node.args[0], ast_Constant_Type
                    ):  # spacename.reverse('constant')
                        result = self._replacement(
                            ast_String_value(node.args[0]),
                            node.func.ctx,
                            None,
                            transpose_lead=True,
                        )
        # handle clip as a method
        if isinstance(node.func, ast.Attribute) and node.func.attr == "clip":
            if isinstance(node.func.value, ast.Name) and node.func.value.id == "np":
                # call to np.clip(...), change to local clip implementation
                clip_args = []
                for a in node.args:
                    clip_args.append(self.visit(a))
                result = ast.Call(
                    func=ast.Name("clip", cts=ast.Load()),
                    args=clip_args,
                    keywords=[self.visit(i) for i in node.keywords],
                )
            elif len(node.args) == 1 and len(node.keywords) == 0:
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
                            ast_String_value(node.args[0]), node.func.ctx, None
                        )
                        backward = self._replacement(
                            ast_String_value(node.args[0]),
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
        # change `x.isin([2,3,4])` to `x == 2 or x == 3 or x == 4`
        if isinstance(node.func, ast.Attribute) and node.func.attr == "isin":
            ante = self.visit(node.func.value)
            targets = node.args
            elts = None
            if len(targets) == 1 and isinstance(targets[0], (ast.List, ast.Tuple)):
                elts = targets[0].elts
            elif len(targets) == 1 and isinstance(targets[0], ast.Name):
                extra_val = self.extra_vars.get(targets[0].id, None)
                if isinstance(extra_val, (list, tuple)):
                    elts = [ast_Constant(i) for i in extra_val]
            if elts is not None:
                ors = []
                for elt in elts:
                    ors.append(
                        ast.Compare(
                            left=ante, ops=[ast.Eq()], comparators=[self.visit(elt)]
                        )
                    )
                result = self.visit(ast.BoolOp(op=ast.Or(), values=ors))
        # change `x.between(a,b)` to `(a <= x) & (x <= b)`
        if isinstance(node.func, ast.Attribute) and node.func.attr == "between":
            ante = self.visit(node.func.value)
            targets = node.args
            if len(targets) == 2:
                lb, ub = targets
                left = ast.Compare(
                    left=ante, ops=[ast.GtE()], comparators=[self.visit(lb)]
                )
                right = ast.Compare(
                    left=ante, ops=[ast.LtE()], comparators=[self.visit(ub)]
                )
                result = ast.BinOp(left=left, op=ast.BitAnd(), right=right)

        # change XXX.isna() [with no arguments] to np.isnan(x)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "isna"
            and len(node.args) == 0
            and len(node.keywords) == 0
        ):
            apply_args = [self.visit(node.func.value)]  # convert XXX into argument
            result = ast.Call(
                func=ast.Name("isnan_fast_safe"),
                args=apply_args,
                keywords=[],
            )
            # if the value XXX is a categorical, rather than just `np.isnan` we also
            # want to check if the category number is less than zero
            if isinstance(apply_args[0], ast.Subscript):
                if isinstance(apply_args[0].value, ast.Name):
                    if apply_args[0].value.id.startswith("__encoding_dict__"):
                        cat_is_lt_zero = ast.Compare(
                            left=apply_args[0].slice,
                            ops=[ast.Lt()],
                            comparators=[ast.Num(0)],
                        )
                        result = ast.BoolOp(
                            op=ast.Or(), values=[cat_is_lt_zero, result]
                        )

        # implement x.get("y",z) where x is the spacename
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.func.value.id == self.spacename
        ):
            if len(node.args) == 2 and len(node.keywords) == 0:
                try:
                    result = self._replacement(
                        ast_String_value(node.args[0]), node.func.value.ctx, node
                    )
                except KeyError:
                    if self.get_default:
                        result = self.visit(node.args[1])
                    else:
                        raise
            if (
                len(node.args) == 1
                and len(node.keywords) == 1
                and "default" == node.keywords[0].arg
            ):
                try:
                    result = self._replacement(
                        ast_String_value(node.args[0]), node.func.value.ctx, node
                    )
                except KeyError:
                    if self.get_default:
                        result = self.visit(node.keywords[0].value)
                    else:
                        raise
            if len(node.args) == 1 and len(node.keywords) == 0:
                result = self._replacement(
                    ast_String_value(node.args[0]), node.func.value.ctx, node
                )

        # change np.where(x, y, z) to (y if x else z), for performance reasons
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "where"
            and isinstance(node.func.value, ast.Name)
            and (node.func.value.id == "np" or node.func.value.id == "numpy")
            and len(node.args) == 3
        ):
            if len(node.args) == 3 and len(node.keywords) == 0:
                result = ast.IfExp(
                    test=self.visit(node.args[0]),
                    body=self.visit(node.args[1]),
                    orelse=self.visit(node.args[2]),
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

    def visit_Compare(self, node):
        result = None

        # devolve XXX ==/!= YYY when XXX or YYY is a categorical and the other is a constant category value
        if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.NotEq)):
            left = self.visit(node.left)
            right = self.visit(node.comparators[0])
            left_is_categorical = (
                isinstance(left, ast.Subscript)
                and isinstance(left.value, ast.Name)
                and left.value.id.startswith("__encoding_dict__")
            )
            if left_is_categorical and isinstance(right, ast.Constant):
                # left is categorical, right is a constant
                left_spacename = left.value.id.split("__")[2]
                left_varname = left.value.id.split("__")[3]
                if (
                    left_spacename == self.spacename
                    and left_varname in self.digital_encodings
                ):
                    left_dictionary = self.digital_encodings[left_varname].get(
                        "dictionary", np.atleast_1d([])
                    )
                    try:
                        right_decoded = np.where(left_dictionary == right.value)[0][0]
                    except IndexError:
                        right_decoded = None
                        warnings.warn(
                            f"right hand value {right.value!r} not found in "
                            f"categories for {left_varname} in {self.spacename}"
                            f"\nexpression: {self.original_expr}"
                            f"\ncategories: {left_dictionary}",
                            stacklevel=2,
                        )
                        # at this point, the right value is not in the left's categories, so
                        # it is guaranteed to be not equal to any of the categories.
                        if isinstance(node.ops[0], ast.Eq):
                            result = ast.Constant(False)
                        elif isinstance(node.ops[0], ast.NotEq):
                            result = ast.Constant(True)
                        else:
                            raise ValueError(
                                f"unexpected operator {node.ops[0]}"
                            ) from None
                    if right_decoded is not None:
                        result = ast.Compare(
                            left=left.slice,
                            ops=[self.visit(i) for i in node.ops],
                            comparators=[ast_Constant(right_decoded)],
                        )
            right_is_categorical = (
                isinstance(right, ast.Subscript)
                and isinstance(right.value, ast.Name)
                and right.value.id.startswith("__encoding_dict__")
            )
            if right_is_categorical and isinstance(left, ast.Constant):
                # right is categorical, left is a constant
                right_spacename = right.value.id.split("__")[2]
                right_varname = right.value.id.split("__")[3]
                if (
                    right_spacename == self.spacename
                    and right_varname in self.digital_encodings
                ):
                    right_dictionary = self.digital_encodings[right_varname].get(
                        "dictionary", np.atleast_1d([])
                    )
                    try:
                        left_decoded = np.where(right_dictionary == left.value)[0][0]
                    except IndexError:
                        left_decoded = None
                        warnings.warn(
                            f"left hand value {left.value!r} not found in "
                            f"categories for {right_varname} in {self.spacename}"
                            f"\nexpression: {self.original_expr}"
                            f"\ncategories: {right_dictionary}",
                            stacklevel=2,
                        )
                        # at this point, the left value is not in the right's categories, so
                        # it is guaranteed to be not equal to any of the categories.
                        if isinstance(node.ops[0], ast.Eq):
                            result = ast.Constant(False)
                        elif isinstance(node.ops[0], ast.NotEq):
                            result = ast.Constant(True)
                        else:
                            raise ValueError(
                                f"unexpected operator {node.ops[0]}"
                            ) from None
                    if left_decoded is not None:
                        result = ast.Compare(
                            left=ast_Constant(left_decoded),
                            ops=[self.visit(i) for i in node.ops],
                            comparators=[right.slice],
                        )

        # if no other changes
        if result is None:
            result = ast.Compare(
                left=self.visit(node.left),
                ops=[self.visit(i) for i in node.ops],
                comparators=[self.visit(i) for i in node.comparators],
            )
        self.log_event("visit_Compare", node, result)
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
    extra_vars=None,
    blenders=None,
    bool_wrapping=False,
    swallow_errors=False,
    get_default=False,
    original_expr=None,
):
    """
    Rewrite an expression so numba can compile it.

    Parameters
    ----------
    expr : str
        The expression being rewritten
    spacename : str
        A namespace of variables that might be in the expression.
    dim_slots : tuple or Any
    spacevars : Mapping, optional
    rawname : str
    rawalias : str
    digital_encodings : Mapping, optional
    prefer_name : str, optional
    extra_vars : Mapping, optional
    blenders : Mapping, optional
    bool_wrapping : bool, optional
    swallow_errors : bool, optional
    get_default : bool, optional
    original_expr : str, optional
        Original (pre-processing) expression, used for debugging

    Returns
    -------
    str
    """
    with warnings.catch_warnings(record=True) as warning_list:
        result = unparse_(
            RewriteForNumba(
                spacename,
                dim_slots,
                spacevars,
                rawname,
                rawalias,
                digital_encodings,
                prefer_name,
                extra_vars,
                blenders,
                bool_wrapping,
                swallow_errors,
                get_default,
                original_expr=original_expr or expr,
            ).visit(ast.parse(expr))
        )
    for warning in warning_list:
        warnings.warn(warning.message, warning.category, stacklevel=2)
    return result


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
            result = unparse_(new_tree)
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
