from functools import wraps


class AliasedValue(tuple):
    def __new__(cls, *args):
        return super().__new__(cls, args)

    def __repr__(self):
        return type(self).__name__ + super().__repr__()


def AliasedValues(*args):
    def iter_():
        for value in args:
            assert not isinstance(
                value, AliasedValue
            ), 'A value tuple inside AliasedValues already implicitly denotes an AliasedValue, no need to express that it is an AliasedValue explicitly'
            yield AliasedValue(*value)

    return tuple(iter_())


class NoDefault:
    pass


class ParameterHelper:
    def __init__(self, specs):
        '''
        Look at example usage in this file.
        '''
        self.alias_to_parameter_names = {}
        self.specs = {}
        for name, data in specs.items():
            self.alias_to_parameter_names[name] = name

            raw_alias = data.get('alias', ())
            if isinstance(raw_alias, str):
                aliases = tuple(raw_alias.split())
            else:
                assert all(isinstance(alias, str) for alias in raw_alias)
                aliases = tuple(raw_alias)
            for alias in aliases:
                assert alias not in self.alias_to_parameter_names
                self.alias_to_parameter_names[alias] = name

            raw_values = data.get('values', ())
            value_list = []
            for value in raw_values:
                if isinstance(value, AliasedValue):
                    value_list.append(value)
                else:
                    value_list.append(AliasedValue(value))
            values = tuple(value_list)

            type_ = data.get('type', object)

            allow = data.get('allow', None)

            default = data.get('default', NoDefault)
            # Default actually has to strictly be a canonical allowed value; it is helpful to avoid mistakes
            if default is not NoDefault:
                flag = False
                if not values and not allow:
                    flag = isinstance(default, type_)
                for value in values:
                    if default == value[0] and issubclass(type(default), type(value[0])):
                        flag = True
                        break
                else:
                    if isinstance(default, type_) and allow and allow(default):
                        flag = True
                assert flag, f"Provided default {default!r} is not an allowed value of parameter '{name}'"

            self.specs[name] = dict(
                aliases=aliases,
                values=values,
                type=type_,
                allow=allow,
                default=default,
            )

    def __call__(self, function):
        '''Wrap a function to specify and handle its parameters.'''

        @wraps(function)
        def wrapper(**parameters):
            canonical = self.convert_to_canonical(**parameters)
            for name, data in self.specs.items():
                if name not in canonical:
                    assert data['default'] is not NoDefault, f"'{name}' parameter is required"
                    canonical[name] = data['default']
            return function(parameters, **canonical)

        return wrapper

    def convert_to_canonical(self, **parameters):
        return {
            self.get_canonical_name(name): self.get_canonical_value(self.get_canonical_name(name), value)
            for name, value in parameters.items()
        }

    def convert_to_alias(self, **parameters):
        return {
            self.get_alias_name(name): self.get_alias_value(self.get_canonical_name(name), value)
            for name, value in parameters.items()
        }

    def get_canonical_name(self, parameter_alias):
        assert (
            parameter_alias in self.alias_to_parameter_names
        ), 'Given parameter_alias does not match any alias for a parameter name'
        return self.alias_to_parameter_names[parameter_alias]

    def get_alias_name(self, parameter_alias):
        assert (
            parameter_alias in self.alias_to_parameter_names
        ), 'Given parameter_alias does not match any alias for a parameter name'
        name = self.alias_to_parameter_names[parameter_alias]
        data = self.specs[name]
        return data['aliases'][0] if data['aliases'] else name

    def get_canonical_value(self, parameter_name, value):
        data = self.specs[parameter_name]
        if not data['values'] and not data['allow'] and isinstance(value, data['type']):
            return value
        for aliases in data['values']:
            if any(value == alias and issubclass(type(value), type(alias)) for alias in aliases):
                return aliases[0]
        if isinstance(value, data['type']) and data['allow'] and data['allow'](value):
            return value
        assert False, 'Given value not one of the allowed ones'

    def get_alias_value(self, parameter_name, value):
        data = self.specs[parameter_name]
        if not data['values'] and not data['allow'] and isinstance(value, data['type']):
            return value
        for aliases in data['values']:
            if any(value == alias and issubclass(type(value), type(alias)) for alias in aliases):
                return aliases[1] if len(aliases) >= 2 else aliases[0]
        if isinstance(value, data['type']) and data['allow'] and data['allow'](value):
            return value
        assert False, 'Given value not one of the allowed ones'


if __name__ == '__main__':
    specs = dict(
        my_param=dict(
            type=str,
            values=('one_option', AliasedValue('another_option', 'an_alias')),
            default='one_option',  # Must be an allowed canonical value
        ),
        another_param=dict(
            type=int | float,  # tuple of types works as well as union
            allow=lambda another_param: another_param >= 0,
            values=(
                AliasedValue(-1, 'invalid'),  # Input type has to be subclass of the given value here too to match
            ),  # The values are allowed in addition to what's allowed by allow setting
            # No default means it is required
        ),
        final_param_canonical_name=dict(
            alias=('fp',),  # Can also be just one string, or a space separated string
            values=AliasedValues(
                ('canonical_long_value', 'clv', 0, 'zero'),
                ('canonical_very_long_possible_value', 'cvlpv', 1),
            ),
            default='canonical_long_value',  # The default value is used exactly as is if parameter is not provided
        ),
    )
    parameter_helper = ParameterHelper(specs)

    @parameter_helper
    def function(parameters, /, *, my_param, another_param, final_param_canonical_name):
        print(f'{parameters=}')
        print(f'{my_param=}')
        print(f'{another_param=}')
        print(f'{final_param_canonical_name=}')

    print(f'{function(my_param='one_option', another_param=5, fp='clv') = }')

    print(f'{function(another_param='invalid') = }')

    print(f'{function(my_param='an_alias', another_param=-1) = }')
