import collections as _collections

from itertools import chain
from importlib import reload
from functools import wraps, reduce


#TODO change self.__class__ to type(self) and other magic accesses
# like self.__magicattr__ to super(object, self).__magicattr__ so that
# if self overrides __getattribute__ these are not broken.
# Hell maybe even do super(type, type(self)).__magicmethod__(self) to
# defend against custom behaviour defined in type(self)


class Singleton(type):
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


def closure(f):
    '''decorator that immediately calles the decorated callable with no arguments'''
    return f()

def inside(cls, name=None):
    '''decorator with one argument that sets the decorated as an attribute of the argument. also optional argument `name` exists.'''
    def put_inside(a):
        nonlocal name
        name = name if name is not None else a.__name__
        setattr(cls, name, a)
        return a
    return put_inside


def itemsetter(*items):
    if len(items) == 1:
        item = items[0]
        def g(obj, value):
            obj[item] = value
    else:
        def g(obj, values):
            for item, value in zip(items, values):
                obj[item] = value
    return g


class Stop(BaseException):
    '''special exception to indicate STOP when doing interactive work in notebook'''
    pass


class Cache(dict):
    __slots__ = ('function',)

    def __init__(self, function):
        self.function = function
    
    def __getitem__(self, args):
        if not isinstance(args, tuple):
            args = (args,)
        if args not in self:
            super().__setitem__(args, self.function(*args))
        return super().__getitem__(args)

    def __setitem__(self, *args, **kwargs):
        raise TypeError("Cannot set value to Cache object")


def memoized(function):
    '''Use as decorator
    DOES NOT SUPPORT kwargs for function! Will ignore all kwargs.
    '''
    cache = Cache(function)
    @wraps(function)
    def memoized(*args):
        return cache[args]
    memoized.cache = cache
    return memoized

def memo(function):
    cache = {}
    @wraps(function)
    def memoized_function(*args, **kwargs):
        try:
            entry = (args, tuple(sorted(kwargs.items())))
            if entry in cache:
                return cache[entry]
            else:
                result = function(*args, **kwargs)
                cache[entry] = result
                return result
        except TypeError:
            raise TypeError("memoized function cannot take unhashable arguments")
        return function(*args, **kwargs)
    memoized_function.cache = cache
    return memoized_function


def multiline_code(codestr):
    rc = _re.compile(r'\n\s*')
    indent = rc.match(codestr).group()
    return codestr.replace(indent, '\n')

    
def compose(*args, l2r=True, r2l=False):
    funclist = list(args)
    if r2l:
        funclist.reverse()
    def composed_function(*args, **kwargs):
        res = funclist[0](*args, **kwargs)
        for f in funclist[1:]:
            res = f(res)
        return res
    return composed_function


def merged_dict(*dicts):
    if len(dicts) == 1 and isinstance(dicts[0], _types.GeneratorType):
        dicts = dicts[0]
    ret = {}
    for d in dicts:
        ret.update(d)
    return ret


def dictmap(function, mapping):
    return {key: function(val) for key, val in type(mapping).items(mapping)}


def frozen_mapping(mapping):
    return tuple(sorted(mapping.items()))


# math/functional
def identity(x):
    return x

def prod(iterable, start=1):
    product = start
    for x in iterable:
        product *= x
    return product

def binom(n,k):
    k = min(k, n - k)
    return reduce(lambda a, b: a * (n - b) // (b + 1), range(k), 1)

def adder(left=None, right=None):
    if left is not None:
        return lambda x: left + x
    elif right is not None:
        return lambda x: x + right
    else:
        raise TypeError("operation only takes one of left or right")

def scaler(left=None, right=None):
    if left is not None:
        return lambda x: left * x
    elif right is not None:
        return lambda x: x * right
    else:
        raise TypeError("operation only takes one of left or right")

def power(exponent):
    return lambda x: x ** exponent

