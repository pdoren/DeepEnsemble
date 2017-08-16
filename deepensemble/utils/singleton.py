__all__ = ['Singleton']

class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]

# noinspection PyMissingConstructor,PyClassHasNoInit
class Singleton(_Singleton('SingletonMeta', (object,), {})):
    """ Singleton class (pattern of design)"""
    pass