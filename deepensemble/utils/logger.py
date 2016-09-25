import sys
import time

__all__ = ['Logger']


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


# noinspection PyMissingConstructor,PyClassHasNoInit
class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


# noinspection PyMissingConstructor
class Logger(Singleton):
    def __init__(self):
        self.log_activate = True
        self.tic = 0
        self.toc = 0

    def print(self, message="", **kwargs):
        if self.log_activate:
            print(str(message), **kwargs)

    def start_measure_time(self, message="", **kwargs):
        self.tic = time.time()
        self.print(message="%s" % message, end="", **kwargs)

    def stop_measure_time(self, message="", **kwargs):
        self.toc = time.time()
        self.print(message=" %s - elapsed: %.2f [s]" % (message, self.toc - self.tic), **kwargs)

    def progressbar_training(self, max_epoch, model, **kwargs):
        return self.progressbar(it=range(0, max_epoch), prefix="%s - epoch:" % model.get_name(), **kwargs)

    def progressbar(self, it, prefix="", postfix="", end="", size=20):
        count = len(it)

        def _show(_i):
            x = int(size * _i / count)
            if _i == 1:
                self.tic = time.time()
            self.toc = time.time()
            dt = self.toc - self.tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            sys.stdout.write(s)
            sys.stdout.flush()

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        sys.stdout.write("%s\n" % end)
        sys.stdout.flush()

    def create_table(self):
        pass
