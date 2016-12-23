from __future__ import print_function

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

    tic : list[]
        Save init value timer.

    buffer : str
        Buffer for save all print text.
    """

    def __init__(self):
        self.log_activate = True
        self.tic = []
        self.buffer = ""
        self.fold = [0]

    def reset(self):
        """ Reset intern parameters.

        """
        self.tic = []
        self.buffer = ""
        self.fold[-1] = 0

    def is_log_activate(self):
        return self.log_activate

    def log_enable(self):
        """ Enable print on console.

        """
        self.log_activate = True

    def log_disable(self):
        """ Disable print on console.

        """
        self.log_activate = False

    # noinspection PyUnusedLocal
    def push_buffer(self, message, end='\n', **kwargs):
        """ Push message in buffer.

        Parameters
        ----------
        message : str
            Message.

        end : str
            This string is concatenate in the end of message.

        kwargs
        """
        self.buffer += message + end

    def get_buffer(self):
        """ Gets buffer.

        Returns
        -------
        str
            Returns the buffer.
        """
        return self.buffer

    def save_buffer(self, filename):
        """ Save the buffer in a file.

        Parameters
        ----------
        filename : str
            Name of file.
        """
        file_ = open(filename, 'w')
        file_.write(self.buffer)
        file_.close()

    def log(self, message="", **kwargs):
        """ Print message in console, also the message is saved in the buffer.

        Parameters
        ----------
        message : str
            String to show in console.

        kwargs
        """
        if self.log_activate:
            print(message, **kwargs)
        self.push_buffer(message, **kwargs)

    # noinspection PyUnusedLocal
    def write(self, message="", write_buf=False, **kwargs):
        """ Write message in console and the buffer.

        Parameters
        ----------
        message : str
            Message.

        write_buf : bool
            Flag for indicate if the message is copied in buffer.

        kwargs
        """
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
        self.tic.append(time.time())
        self.log(message="%s" % message, end="", **kwargs)

    def stop_measure_time(self, message="", **kwargs):
        """ Stop timer, also show a message with the time elapsed since it called the start_measure_time function.

        Parameters
        ----------
        message : str
            Message that it want to print in console.

        kwargs
        """
        tic = self.tic.pop()
        toc = time.time()
        self.log(message=" %s - elapsed: %.2f [s]" % (message, toc - tic), **kwargs)

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
        size = 20
        tic = time.time()

        self.fold[-1] += 1

        def _show(_i):
            if _i == 0:
                self.fold.append(1)

            prefix = "%s - fold: %d, epoch:" % (model.get_name(), self.fold[-1])
            postfix = "| score: %.4f / %.4f" % (model.get_train_score(), model.get_test_score())
            x = int(size * _i / count)
            toc = time.time()
            dt = toc - tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("\n")
        self.fold.pop()

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
        size = 20
        tic = time.time()

        self.fold[-1] += 1

        def _show(_i):
            if _i == 0:
                self.fold.append(1)

            prefix = "%s - fold: %d, epoch:" % (model.get_name(), self.fold[-1])
            postfix = "| score: %.4f / %.4f" % (model.get_train_score(), model.get_test_score())
            x = int(size * _i / count)
            toc = time.time()
            dt = toc - tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("\n")
        self.fold.pop()

    def progressbar(self, it, prefix="", postfix="", end="", size=20):
        """ Show a progressbar (it is necessary called for increment counter).

        Parameters
        ----------
        it : iterator or list[]
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
        tic = time.time()

        def _show(_i):
            x = int(size * _i / count)
            toc = time.time()
            dt = toc - tic
            eta = (count - _i) * dt / (_i + 1)  # Estimate Time Arrival
            s = "\r%s[%s%s] %i/%i elapsed: %.2f[s] - left: %.2f[s] %s" % (
                prefix, "#" * x, "." * (size - x), _i, count, dt, eta, postfix)
            self.write(s)

        _show(0)
        for i, item in enumerate(it):
            yield item
            _show(i + 1)
        self.write("%s\n" % end)
