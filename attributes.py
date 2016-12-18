########################################################################
#                         ATTRIBUTE ACCESSING                          #
########################################################################

class _Flag():
    pass
_flag = _Flag()

class Attribute(str):
    _aa_insts = {}
    def __new__(cls, name):
        try:
            return cls._aa_insts[cls, name]
        except KeyError:
            inst = super().__new__(cls, name)
            cls._aa_insts[cls, name] = inst
            return inst

    def __repr__(self):
        return "{{.{}}}".format(self)

    def __call__(self, obj=None, set=_flag, dlt=_flag):
        if dlt is not _flag:
            return delattr(obj, self)
        if set is not _flag:
            return setattr(obj, self, set)
        else:
            return getattr(obj, self)

class _AttributeAccess():
    def __getattr__(self, attr):
        return Attribute(attr)

    def __call__(self, *args):
        def gen_aa_fn(obj):
            for a in args:
                yield a(obj)
        return gen_aa_fn

    def __getitem__(self, args):
        def list_aa_fn(obj):
            return [a(obj) for a in args]
        return list_aa_fn


attribute_accessor = _AttributeAccess()
