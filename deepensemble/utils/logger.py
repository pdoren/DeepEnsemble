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
    """ Singleton class (pattern of design)"""
    pass


# noinspection PyMissingConstructor
class Logger(Singleton):
    """ Class for controlling log and console messages.

    Attributes
    ----------
    log_activate : bool, True by default
        Flag for print or not text in console.

    tic : long
        Save init value timer.

    toc : long
        Save end value timer.

    buffer : str
        Buffer for save all print text.
    """

    def __init__(self):
        self.log_activate = True
        self.tic = 0
        self.toc = 0
        self.buffer = ""
        self.fold = 0

    def reset(self):
        self.tic = 0
        self.toc = 0
        self.buffer = ""
        self.fold = 0

    def log_enable(self):
        self.log_activate = True

    def log_disable(self):
        self.log_activate = False

    # noinspection PyUnusedLocal
    def push_buffer(self, message, end='\n', **kwargs):
        self.buffer += message + end

    def get_buffer(self):
        return self.buffer

    def save_buffer(self, filename):
        file_ = open(filename, 'w')
        file_.write(self.buffer)
        file_.close()

    def print(self, message="", **kwargs):
        """ Print message in console, also the message is saved in the buffer.

        Parameters
        ----------
        message : str
            String to show in console.

        kwargs
        """
        if self.log_activate:
            print(str(message), **kwargs)
        self.push_buffer(message, **kwargs)

    # noinspection PyUnusedLocal
    def write(self, message="", write_buf=False, **kwargs):
        if self.log_activate:
            sys.stdout.write(message)
            sys.stdout.flush()
            if write_buf:
                self.push_buffer(message, end='')

    def start_measure_time(self, message="", **kwargs):
        """ Start timer, is possible show a message.

        Parameters
        ----------
        message : str
            String to show in console.

        kwargs
        """
        self.tic = time.time()
        self.print(message="%s" % message, end="", **kwargs)

    def stop_measure_time(self, message="", **kwargs):
        """ Stop timer, also show a message with the time elapsed since it called the start_measure_time function.

        Parameters
        ----------
        message : str
            Message that it want to print in console.

        kwargs
        """
        self.toc = time.time()
        self.print(message=" %s - elapsed: %.2f [s]" % (message, self.toc - self.tic), **kwargs)

    def progressbar_training(self, max_epoch, model):
        """ Show a progressbar (it is necessary called for increment counter).

        Parameters
        ----------
        max_epoch : int
            Max epoch or count of progress bar.

        model : Model
            Model used of training.

        Returns
        -------
        iterator
            Returns a iterator each time is called.
        """

        it = range(0, max_epoch)
        count = len(it)
        self.fold += 1
        prefix = "%s - fold: %d, epoch:" % (model.get_name(), self.fold)
        size = 20

        def _show(_i):
            postfix = "| score: %.4f / %.4f" % (model.get_train_score(), model.get_test_score())
            x = int(size * _i / count)
            if _i == 1:
                self.tic = time.time()
            self.toc = time.time()
            dt = self.toc - self.tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("\n")

    def progressbar_training2(self, it, model):
        """ Show a progressbar (it is necessary called for increment counter).

        Parameters
        ----------
        it
            Iterator for progressbar.

        model : Model
            Model used of training.

        Returns
        -------
        iterator
            Returns a iterator each time is called.
        """
        count = len(it)
        self.fold += 1
        prefix = "%s - fold: %d, epoch:" % (model.get_name(), self.fold)
        size = 20

        def _show(_i):
            postfix = "| score: %.4f / %.4f" % (model.get_train_score(), model.get_test_score())
            x = int(size * _i / count)
            if _i == 1:
                self.tic = time.time()
            self.toc = time.time()
            dt = self.toc - self.tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("\n")

    def progressbar(self, it, prefix="", postfix="", end="", size=20):
        """ Show a progressbar (it is necessary called for increment counter).

        Parameters
        ----------
        it : iterator
            Range of values that progressbar uses.

        prefix : str
            This string is displayed before of progressbar.

        postfix : str
            This string is displayed after of progressbar.

        end : str
            This string will be displayed when the count of progressbar ends.

        size : int
            Size of progressbar.
        """
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
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("%s\n" % end)
