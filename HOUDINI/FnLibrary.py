from typing import Dict, Optional
import torch.nn as nn
import pickle
import os.path
import torch
from torch import split

from HOUDINI.Synthesizer.AST import *
from HOUDINI.Synthesizer.AST import PPSort
from HOUDINI.Synthesizer.GenUtils import createDir

"""
t = PPSortVar('T')
t1 = PPSortVar('T1')
t2 = PPSortVar('T2')
"""


class PPLibItem(NamedTuple('PPLibItem', [('name', str), ('sort', PPSort), ('obj', object)])):
    pass


class FnLibrary:
    def __init__(self):
        self.items: Dict[str, PPLibItem] = {}
        # self._add_python_fns()

    def save(self, location):
        """
        Saves the neural networks if already not on disk.
        Saves the name-sort dictionary to lib.pickle
        """

        def isNN(anObj):
            return issubclass(type(anObj), nn.Module)

        if not os.path.exists(location):
            createDir(location)

        for name, li in self.items.items():
            if isNN(li.obj):
                nnFileName = location + '/' + name + '.pth'
                if not os.path.isfile(nnFileName):
                    createDir(location)
                    li.obj.save(location)
            else:
                pass

        newDict = {}
        for name, li in self.items.items():
            newDict[name] = (li.sort, isNN(li.obj))

        with open(location + '/lib.pickle', 'wb') as fh:
            pickle.dump(newDict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def save1(self, location, id):
        """
        Saves the neural networks if already not on disk.
        Saves the name-sort dictionary to lib.pickle
        """

        def isNN(anObj):
            return issubclass(type(anObj), nn.Module)

        for name, li in self.items.items():
            if isNN(li.obj):
                nnFileName = location + '/' + name + '.pth'
                if not os.path.isfile(nnFileName):
                    createDir(location)
                    li.obj.save(location)
            else:
                pass

        newDict = {}
        for name, li in self.items.items():
            newDict[name] = (li.sort, isNN(li.obj))

        with open(location + '/lib{}.pickle'.format(id), 'wb') as fh:
            pickle.dump(newDict, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def addItem(self, libItem: PPLibItem):
        self.items[libItem.name] = libItem
        self.__dict__[libItem.name] = libItem.obj

    def addItems(self, libItems: List[PPLibItem]):
        for li in libItems:
            self.addItem(li)

    def removeItems(self, names: List[str]):
        for name in names:
            self.items.pop(name, None)
            self.__dict__.pop(name, None)

    def get(self, name: str) -> Optional[PPLibItem]:
        res = None
        if name in self.items:
            res = self.items[name]
        return res

    def getWithLibPrefix(self, name: str) -> Optional[PPLibItem]:
        res = None
        if name in self.items:
            res = self.items[name]
            res = PPLibItem('lib.' + res.name, res.sort, res.obj)
        return res

    def set(self, name: str, obj: callable):
        if name in self.items:
            self.items[name].obj = obj

    def getDict(self) -> Dict[str, PPLibItem]:
        return self.items.copy()
    def map2d(fn, iterable):
        if type(iterable) == tuple:
            iterable = iterable[1]

        if isinstance(iterable, torch.autograd.variable.Variable):
            #iterable = split(iterable)
            iterable = [split(i,1) for i in iterable]

        result = [[fn(j) for j in i] for i in iterable]
        return result


    def map_g(fn):
        def ret(iterable):
            if type(iterable) == tuple:
                iterable = iterable[1]

            if isinstance(iterable, torch.autograd.variable.Variable):
                #iterable = split(iterable)
                iterable = [split(i,1) for i in iterable]

            result = [[fn(j) for j in i] for i in iterable]
            return result
        return ret


    def flatten_2d_list(iterable):
        assert (type(iterable) == list)
        assert (iterable.__len__() > 0)
        assert (type(iterable[0]) == list)
        return [item for innerlist in iterable for item in innerlist]


    def cat(a, b):
        if type(a) == tuple:
            a = a[1]
        if type(b) == tuple:
            b = b[1]

        return torch.cat((a, b), dim=1)


    def map(fn, iterable):
        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
        if iterable.__len__() > 0 and type(iterable[0]) == tuple:
            iterable = [i[1] for i in iterable]
    # using list() to force the map to be evaluated, otherwise it's lazily evaluated
        return list(map(fn, iterable))


    def map_list(fn):
        def ret(iterable):
            if isinstance(iterable, torch.autograd.variable.Variable):
                iterable = split(iterable)
            if iterable.__len__() > 0 and type(iterable[0]) == tuple:
                iterable = [i[1] for i in iterable]
        # using list() to force the map to be evaluated, otherwise it's lazily evaluated
            return list(map(fn, iterable))

        return ret


    def conv_list(fn):
        def ret(iterable):
            if isinstance(iterable, torch.autograd.variable.Variable):
                iterable = split(iterable)
            if iterable.__len__() > 0 and type(iterable[0]) == tuple:
                iterable = [i[1] for i in iterable]

            if iterable.__len__() == 0:
                return []

            if type(iterable[0]) != torch.autograd.variable.Variable:
                raise NotHandledException

        # zero-pad start and end
            zero = torch.zeros_like(iterable[0])
            iterable.insert(0, zero)
            iterable.append(zero)
            result = []
            for idx in range(1, iterable.__len__() - 1):
                c_arr = [iterable[idx - 1], iterable[idx], iterable[idx + 1]]
                result.append(fn(c_arr))
            return result

        return ret


    def conv_graph(fn):
        return lambda x: fn(x)


    def reduce_graph(fn):
        raise NotImplementedError


    def reduce_list(fn, init=None):
        def ret(iterable):
            if isinstance(iterable, torch.autograd.variable.Variable):
                iterable = split(iterable)
            if iterable.__len__() > 0 and type(iterable[0]) == tuple:
                iterable = [i[1] for i in iterable]
            return reduce(fn, iterable) if init is None else reduce(fn, iterable, init)

        return ret


    def repeat(num_times_to_repeat, fn):
        def ret(x):
            for i in range(num_times_to_repeat):
                x = fn(x)
            return x

        return ret


    def compose(g, f):
        return lambda x: g(f(x))


    def reduce(fn, iterable, initializer=None):
        if isinstance(iterable, torch.autograd.variable.Variable):
            iterable = split(iterable)
        if iterable.__len__() > 0 and type(iterable[0]) == tuple:
            iterable = [i[1] for i in iterable]
        return reduce(fn, iterable) if initializer is None else reduce(fn, iterable, initializer)


    def get_zeros(dim):
        """
    Returns zeros of shape [var_x.shape[0], dim]
    Atm, it's used for initializing hidden state
    :return:
        """
        """
    if type(var_x) == tuple:
        var_x = var_x[1]
        """

        zeros = torch.zeros(1, dim)
        if torch.cuda.is_available():
            zeros = zeros.cuda()
        return torch.autograd.variable.Variable(zeros)


    def multiply_by_range09():
    # zeros = torch.zeros(var_x.data.shape[0], dim)
        range_np = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.], dtype=np.float32).reshape((10, 1))
        range_torch = torch.from_numpy(range_np)
        if torch.cuda.is_available():
            range_torch = range_torch.cuda()
        range_var = torch.autograd.variable.Variable(range_torch)

    def multiply_by_range09(inputs):
        return torch.matmul(inputs, range_var)

        return pp_multiply_by_range09


    def argmax(x):
    # x = list(x)
        if type(x) == tuple:
            x = x[1]
        values, indices = torch.max(x, 1, keepdim=True)
    # values, indices = torch.max(input=x, dim=1, keepdim=True)
    # return indices
        indices = indices.float()
        return indices


    def add(x, y):
        if type(x) == tuple:
            x = x[1]
        if type(y) == tuple:
            y = y[1]
        return x + y