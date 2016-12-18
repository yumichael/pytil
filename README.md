## Python ease-of-use add-on utility modules

Typical boilerplate code I would add to projects is
```python
from pytil.utility import *
from pytil.object import Namespace as Ob               # optional include
from pytil.attributes import attribute_accessor as AA  # optional include
```

#### pytil.utility
Importing this using `import *` imports some things hidden away in Python's standard library package into the global namespace
and adds some of my own common metaclasses, functional programming helpers, and various other things.

#### pytil.object
This module contains the `Namespace` class, which has the internal name `<>` and goes by the idiomatic referencee `Ob`.
`Namespace` is a subclass of `dict` that routes all attributes to dictionary lookup.
So it is like a Javascript object, except its methods cannot be accessed by conventional `obj.method` notation
as that would instead mean `obj['method']`. It has most of the expected magic methods implemented,
_e.g._ `obj.foo = 58` translates to `obj['foo'] = 58`.

#### pytil.attributes
This module contains a class `Attribute` that subclasses `str` whose instances are meant to represent attribute access with a fixed name.
An instance of `Attribute` is callable, and its action on `x` is to return a specific attribute of `x`.
This module also includes `attribute_accessor`, idiomatically imported as `AA`, which lets you get easy access to Attribute instances
by the way of `AA.some_name is Attribute('some_name')` being true.

Here are examples.
```python
>>> from pytil.object import Namespace as Ob
>>> thing = Ob(hello=35)
>>> thing
<>(hello=35)
>>> thing.what = 72
>>> thing
<>(hello=35, what=72)
>>> from pytil.attributes import attribute_accessor as AA
>>> AA.what(thing)
72
>>> # you can also create an Ob using class definition syntax
>>> class another_thing(Ob()):
...     hello = 88
...
>>> another_thing
<>(hello=88)
```
