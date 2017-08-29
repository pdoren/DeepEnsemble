from .singleton import Singleton

__all__ = ['TextTranslation']


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

        with open(dir + "/languages/dict_language_%s.txt" % self.current_language, encoding="utf-8") as f:
            for line in f:
                (key, val) = line.split(':', 1)
                self.dict_trans[key.strip()] = val.strip()

    def set_current_language(self, name_language):
        self.current_language = name_language
        self._load_language()


