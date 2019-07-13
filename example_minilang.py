"""
Define a mini-language to compute the fibonnaci sequence.

Uses LLVM for codegeneration.
"""

import inspect
from llvmlite import ir
from llvmlite import binding as llvm

from kvbacker import (
    ResourceManager,
    Managed,
    ManagedList,
    graphviz_render_revisions,
)


class FunctionCollections(Managed):
    pass


class ContextResources(ResourceManager):
    pass


class Expr(Managed):
    pass


class FunctionDecl(Expr):
    def init(self, name):
        self.name = name

    def __call__(self, *args):
        args = self._resmngr.new(
            ArgListVal,
            [_fix_value(self._resmngr, x) for x in args],
        )
        return self._resmngr.new(CallVal, op=self, args=args)

    def codegen(self, cgstate):
        mod = cgstate.builder.module
        return mod.get_global(self.name)


class FunctionDefn(Expr):
    def init(self, name, expr, arity):
        self.name = name
        self.expr = expr
        self.arity = arity

    def codegen_definition(self, ir_mod):
        fn = self.codegen_declare(ir_mod)
        fn.calling_convention = 'fastcc'
        entry_block = fn.append_basic_block('entry')
        main_block = fn.append_basic_block('main')
        irbuilder = ir.IRBuilder()
        cgstate = CodegenState(irbuilder, entry_block)
        irbuilder.position_at_end(main_block)
        value = self.expr.codegen(cgstate)
        irbuilder.ret(value)
        # Clean up
        irbuilder.position_at_end(entry_block)
        irbuilder.branch(main_block)

    def codegen_declare(self, ir_mod):
        try:
            return ir_mod.get_global(self.name)
        except KeyError:
            argty = ir.IntType(32)
            fnty = ir.FunctionType(argty, [argty] * self.arity)
            fn = ir.Function(ir_mod, fnty, name=self.name)
            return fn


class ConstVal(Expr):
    def init(self, value):
        self.value = value

    def codegen(self, cgstate):
        intty = ir.IntType(32)
        return intty(self.value)


class ParamVal(Expr):
    def init(self, name, pos):
        self.name = name
        self.pos = pos

    def codegen(self, cgstate):
        builder = cgstate.builder
        fn = builder.function
        arg = fn.args[self.pos]
        return arg


class CallVal(Expr):
    def init(self, op, args):
        self.op = op
        self.args = args

    def codegen(self, cgstate):
        if isinstance(self.op, Expr):
            assert isinstance(self.op, FunctionDecl)
            callee = self.op.codegen(cgstate)
            args = [a.codegen(cgstate) for a in self.args]
            return cgstate.builder.call(callee, args)
        else:
            builder = cgstate.builder
            assert len(self.args) == 2
            lhs = self.args[0].codegen(cgstate)
            rhs = self.args[1].codegen(cgstate)
            if self.op == '+':
                res = builder.add(lhs, rhs)
            elif self.op == '-':
                res = builder.sub(lhs, rhs)
            elif self.op == '>':
                res = builder.icmp_signed('>', lhs, rhs)
            elif self.op == '==':
                res = builder.icmp_signed('==', lhs, rhs)
            else:
                raise NotImplementedError(self.op)
            return res


class ArgListVal(ManagedList):
    pass


class IfElseVal(Expr):
    def init(self, pred, then_expr, else_expr):
        self.pred = pred
        self.then_expr = then_expr
        self.else_expr = else_expr

    def codegen(self, cgstate):
        builder = cgstate.builder
        bb_then = builder.append_basic_block('then')
        bb_else = builder.append_basic_block('else')
        bb_after = builder.append_basic_block('endif')

        pred = self.pred.codegen(cgstate)
        builder.cbranch(pred, bb_then, bb_else)

        builder.position_at_end(cgstate.entry_block)
        phi = builder.alloca(ir.IntType(32))

        builder.position_at_end(bb_then)
        then_value = self.then_expr.codegen(cgstate)
        builder.store(then_value, phi)
        builder.branch(bb_after)

        builder.position_at_end(bb_else)
        else_value = self.else_expr.codegen(cgstate)
        builder.store(else_value, phi)
        builder.branch(bb_after)

        builder.position_at_end(bb_after)
        return builder.load(phi)


class Context:
    def __init__(self):
        self._rm = ContextResources()
        self._declfuncs = {}
        self._definitions = {}

    def define(self, fn):
        fname = fn.__name__
        fndecl = self._rm.new(FunctionDecl, name=fname)
        self._declfuncs[fname] = {'decl': fndecl, 'defn': fn}
        return fndecl

    def visualize(self):
        return self._rm.visualize()

    def codegen(self):
        ir_mod = ir.Module()
        for k, defn in self._definitions.items():
            defn.codegen_declare(ir_mod)
        for k, defn in self._definitions.items():
            defn.codegen_definition(ir_mod)
        return ir_mod

    def materialize(self):
        while self._declfuncs:
            name, info = self._declfuncs.popitem()
            defn = info['defn']
            defn = self._build_definition(name, defn)
            self._definitions[name] = defn

    def _build_definition(self, name, fn):
        sig = inspect.signature(fn)
        params = list(sig.parameters.items())
        kwargs = {params[0][0]: self}
        for i, (k, v) in enumerate(params[1:]):
            kwargs[k] = self._rm.new(ParamVal, name=k, pos=i)
        expr = fn(**kwargs)
        defn = self._rm.new(
            FunctionDefn, name=name, expr=expr, arity=len(params) - 1,
        )
        return defn

    def call(self, op, args):
        return self._rm.new(CallVal, op=op, args=self._arglist(*args))

    def ifelse(self, pred, then_expr, else_expr):
        return self._rm.new(
            IfElseVal,
            pred=pred,
            then_expr=self._fix_value(then_expr),
            else_expr=self._fix_value(else_expr),
        )

    def _arglist(self, *args):
        return self._rm.new(
            ArgListVal,
            values=[self._fix_value(x) for x in args],
        )

    def _fix_value(self, val):
        return _fix_value(self._rm, val)


def _fix_value(rm, val):
    if isinstance(val, Expr):
        return val
    else:
        return rm.new(ConstVal, value=val)


class CodegenState:
    def __init__(self, ir_builder, entry_block):
        self.builder = ir_builder
        self.entry_block = entry_block


def make_c_wrapper(fn_callee):
    mod = fn_callee.module
    fnty = fn_callee.function_type
    fn = ir.Function(mod, fnty, name='entry_' + fn_callee.name)
    builder = ir.IRBuilder(fn.append_basic_block())
    builder.ret(builder.call(fn_callee, fn.args))


def execute(ir_mod):
    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()

    llmod = llvm.parse_assembly(str(ir_mod))

    print('optimized'.center(80, '-'))
    pmb = llvm.create_pass_manager_builder()
    pmb.opt_level = 1
    pm = llvm.create_module_pass_manager()
    pmb.populate(pm)
    pm.run(llmod)
    print(llmod)

    target_machine = llvm.Target.from_default_triple().create_target_machine()

    with llvm.create_mcjit_compiler(llmod, target_machine) as ee:
        ee.finalize_object()
        cfptr = ee.get_function_address("entry_fib")

        from ctypes import CFUNCTYPE, c_int

        cfunc = CFUNCTYPE(c_int, c_int)(cfptr)

        # TEST
        for i in range(12):
            res = cfunc(i)
            print('fib({}) = {}'.format(i, res))

        # Get CFG
        ll_fib_more = llmod.get_function('fib_more')
        cfg = llvm.get_function_cfg(ll_fib_more)
        llvm.view_dot_graph(cfg, view=True)


def test():
    context = Context()

    @context.define
    def fib(ctx, n):
        return fib_more(n, 0, 1)

    @context.define
    def fib_more(ctx, n, a, b):
        pred_cont = ctx.call('>', [n, 1])
        minus1 = ctx.call('-', [n, 1])
        ab = ctx.call('+', [a, b])
        added = fib_more(minus1, b, ab)

        n_eq_1 = ctx.call('==', [n, 1])
        return ctx.ifelse(pred_cont, added,
                          ctx.ifelse(n_eq_1, b, a))

    context.materialize()

    ir_mod = context.codegen()
    context.visualize().render(view=True)

    make_c_wrapper(ir_mod.get_global('fib'))
    print(ir_mod)

    execute(ir_mod)

    rendered = graphviz_render_revisions(context._rm)
    with open('example_minilang.html', 'w') as fout:
        print(rendered, file=fout)


if __name__ == '__main__':
    test()
