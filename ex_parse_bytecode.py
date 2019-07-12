import dis
import inspect
from collections import namedtuple
from pprint import pprint

from kvbacker import (
    ResourceManager,
    Managed,
    ManagedList,
    graphviz_render_revisions,
)


class Translate:
    def __init__(self, fn):
        self._sig = inspect.signature(fn)
        self._bc = dis.Bytecode(fn)
        insts = list(self._bc)
        self._instmap = {inst.offset: inst for inst in insts}
        self._nextpcmap = {a.offset: b.offset
                           for a, b in zip(insts, insts[1:])}
        self._optable = OpTable()

    def run(self):
        env = Env(pc=0, pc_next=self.advance_pc)
        # Set arguments

        state_init = env.res.internal_dict().save()
        for param in self._sig.parameters:
            env.setup_arg(param)
        try:
            while env.running:
                cur_inst = self._instmap[env.pc]
                op = self._optable[cur_inst.opname]
                op(env, cur_inst)
        finally:
            # # GV
            # out = graphviz_render_revisions(
            #     env.res,
            #     since=state_init,
            #     backend='gv',
            #     attrs={
            #         'name_prefix': 'parse_bytecode',
            #     },
            # )
            out = graphviz_render_revisions(env.res, since=state_init, backend='d3')
            print(out)

    def advance_pc(self, pc):
        return self._nextpcmap[pc]


class Env:
    def __init__(self, pc, pc_next):
        self.res = ResourceManager()
        self.pc = 0
        self.running = True
        self._pc_next = pc_next
        self.stack = self.res.new_list()
        self.instructions = self.res.new_list()
        self.varmap = self.res.new(VarMap)

    def setup_arg(self, name):
        self.varmap[name] = self.res.new(Arg, name=name)

    def stack_push(self, val):
        self.stack.append(val)

    def stack_pop(self):
        return self.stack.pop()

    def store(self, val, varname):
        self.varmap[varname] = val

    def load(self, varname):
        return self.varmap[varname]

    def write_inst(self, opclass, **kwargs):
        inst = self.res.new(opclass, **kwargs)
        self.instructions.append(inst)
        return inst

    def write_call(self, callee, args):
        arglist = self.res.new(ArgList)
        arglist.extend(args)
        return self.write_inst(OpCall, callee=callee, args=arglist)

    def write_branch(self, condition, target):
        self.write_inst(OpBranch, condition=condition, target=target)

    def write_jump(self, target):
        self.write_inst(OpJump, target=target)
        self.running = False

    def pc_next(self):
        self.pc = self._pc_next(self.pc)


class VarMap(Managed):
    def __setitem__(self, key, val):
        setattr(self, key, val)

    def __getitem__(self, key):
        return getattr(self, key)


class ArgList(ManagedList):
    pass


class Val(Managed):
    pass


class Arg(Val):
    def init(self, name):
        self.name = name


class Inst(Val):
    pass


class OpConst(Inst):
    def init(self, constant):
        self.constant = constant


class OpSetupLoop(Inst):
    def init(self, end):
        self.end = end


class OpLoadGlobal(Inst):
    def init(self, name):
        self.name = name


class OpCall(Inst):
    def init(self, callee, args):
        self.callee = callee
        self.args = args


class OpBranch(Inst):
    def init(self, condition, target):
        self.condition = condition
        self.target = target


class OpJump(Inst):
    def init(self, target):
        self.target = target


class OpGetIter(Inst):
    def init(self, value):
        self.value = value


class OpNextIter(Inst):
    def init(self, iterator):
        self.iterator = iterator


class OpIterInvalid(Inst):
    def init(self, iterator):
        self.iterator = iterator


class OpIterState(Inst):
    def init(self, iterator):
        self.iterator = iterator


class OpIterValue(Inst):
    def init(self, iterator):
        self.iterator = iterator


class OpInplaceAdd(Inst):
    def init(self, lhs, rhs):
        self.lhs = lhs
        self.rhs = rhs


State = namedtuple("State", ['next_pc'])


class OpTable:
    """Map opname to translation
    """
    def __getitem__(self, opname):
        return getattr(self, opname)

    def LOAD_CONST(self, env, inst):
        val = env.write_inst(OpConst, constant=inst.argval)
        env.stack_push(val)
        env.pc_next()

    def STORE_FAST(self, env, inst):
        val = env.stack_pop()
        env.store(val, inst.argval)
        env.pc_next()

    def LOAD_FAST(self, env, inst):
        varname = inst.argval
        val = env.load(varname)
        env.stack_push(val)
        env.pc_next()

    def SETUP_LOOP(self, env, inst):
        env.write_inst(OpSetupLoop, end=inst.argval)
        env.pc_next()

    def LOAD_GLOBAL(self, env, inst):
        val = env.write_inst(OpLoadGlobal, name=inst.argval)
        env.stack_push(val)
        env.pc_next()

    def CALL_FUNCTION(self, env, inst):
        argct = inst.argval
        args = [env.stack_pop() for _ in range(argct)][::-1]
        callee = env.stack_pop()
        res = env.write_call(callee, args)
        env.stack_push(res)
        env.pc_next()

    def GET_ITER(self, env, inst):
        obj = env.stack_pop()
        res = env.write_inst(OpGetIter, value=obj)
        env.stack_push(res)
        env.pc_next()

    def FOR_ITER(self, env, inst):
        iterator = env.stack_pop()
        advanced = env.write_inst(OpNextIter, iterator=iterator)
        invalid = env.write_inst(OpIterInvalid, iterator=advanced)
        env.write_branch(invalid, inst.argval)

        iterator = env.write_inst(OpIterState, iterator=iterator)
        env.stack_push(iterator)
        value = env.write_inst(OpIterValue, iterator=iterator)
        env.stack_push(value)
        env.pc_next()

    def INPLACE_ADD(self, env, inst):
        rhs = env.stack_pop()
        lhs = env.stack_pop()
        res = env.write_inst(OpInplaceAdd, lhs=lhs, rhs=rhs)
        env.stack_push(res)
        env.pc_next()

    def JUMP_ABSOLUTE(self, env, inst):
        env.write_jump(inst.argval)


def run_compiler(fn):
    translate = Translate(fn)
    translate.run()


def test():
    def ex1_loop(n):
        c = 0
        for i in range(n):
            c += 1
        return c

    run_compiler(ex1_loop)


if __name__ == '__main__':
    test()
