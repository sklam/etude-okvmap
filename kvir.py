from kvbacker import ResourceManager, Managed, ManagedList
from numba.ir import Loc


class IRNode(Managed):
    pass


class Block(ManagedList):
    def init(self, name):
        super(Block, self).init()
        self.name = name


class Statement(IRNode):
    pass


class Assignment(Statement):
    def init(self, target, value, loc):
        self.target = target
        self.value = value
        self.loc = loc


class IRManager(ResourceManager):
    accepted_types = (Loc,)

    def set_entry_block(self, blk):
        self.infos.entry_block = blk

    def get_entry_block(self):
        return self.infos.entry_block


def main():

    loc = Loc('here', 1)

    rm = IRManager()
    blk = rm.new(Block, name='entry')
    blk.append(rm.new(Assignment, target='myvar', value=1, loc=loc))
    blk.append(rm.new(Assignment, target='var2', value=2, loc=loc))
    rm.set_entry_block(blk)

    print(rm.dump())

    print('-' * 40)
    for stmt in blk:
        print(stmt)

    rm2 = rm.clone()

    # Mutate rm
    blk.append(rm.new(Assignment, target='abc', value=blk[0], loc=loc))
    blk[0].value = 321321

    print('-' * 40)
    for stmt in blk:
        print(stmt)

    blk2 = rm2.get_entry_block()
    print('-' * 40)
    for stmt in blk2:
        print(stmt)

    # Replace

    print('-' * 40)

    rm.replace_all(
        blk[0],
        rm.new(Assignment, target='replaced', value=321543, loc=loc),
    )

    for stmt in blk:
        print(stmt)

    print('-' * 40)
    for stmt in blk2:
        print(stmt)

    print('-' * 40)
    print(list(rm.iter_referers(blk[0])))
    print(list(rm.iter_referents(blk[0])))

    g = rm.visualize()
    g.render('kvir.dot', view=True)



if __name__ == '__main__':
    main()
