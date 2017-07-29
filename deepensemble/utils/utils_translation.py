__all__ = ['TextTranslation']

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


# noinspection PyMissingConstructor
class TextTranslation(Singleton):

    def __init__(self):
        self.current_language = 'es'  # Default Language
        self.dict_trans = {}
        self._load_language()

    def get_str(self, name_str):
        return self.dict_trans.get(name_str, '')

    def _load_language(self):
        import os
        dir = os.path.dirname(os.path.abspath(__file__))

        with open(dir + "/languages/dict_language_%s.txt" % self.current_language) as f:
            for line in f:
                (key, val) = line.split(':', 1)
                self.dict_trans[key.strip()] = val.strip()

    def set_current_language(self, name_language):
        self.current_language = name_language
        self._load_language()


