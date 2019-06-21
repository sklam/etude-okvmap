from kvbacker import ResourceManager, Managed, ManagedList
from collections import Sequence


class TypeNode(Managed):
    pass


class Var(TypeNode):
    def init(self, name):
        self.name = name


class Placeholder(TypeNode):
    pass


class TypeInst(TypeNode):
    def init(self, kind, **kwargs):
        self.kind = kind
        for k, v in kwargs.items():
            setattr(self, k, v)


class Kind(TypeNode):
    def init(self, name):
        self.name = name


class FunctionType(ManagedList):
    pass


class TypeSystem(ResourceManager):
    def make_kind(self, name):
        return self.new(Kind, name=name)

    def make_type(self, kind, **kwargs):
        return self.new(TypeInst, kind, bitwidth=32)

    def make_function(self, return_type, *args):
        fnty = self.new(FunctionType)
        fnty.append(return_type)
        for arg in args:
            fnty.append(arg)
        return fnty

    def placeholder(self):
        return self.new(Placeholder)


def main():
    ts = TypeSystem()
    IntegerKind = ts.make_kind('IntegerKind')
    FloatKind = ts.make_kind('FloatKind')

    Int32 = ts.make_type(IntegerKind, bitwidth=32)
    Float = ts.make_type(FloatKind, bitwidth=32)
    print(Int32)

    fn1 = ts.make_function(Int32, Float, Int32)
    print('fn1', fn1)

    ph = ts.placeholder()
    fn2 = ts.make_function(Int32, ph, ph)
    print('fn2', fn2)

    g = ts.visualize()
    g.render('kvir.dot', view=True)
    input("press any key to continue >>")

    ts.replace_all(fn2[1], Int32)
    print('fn2_replaced', fn2)

    g = ts.visualize()
    g.render('kvir.dot', view=True)


if __name__ == '__main__':
    main()
