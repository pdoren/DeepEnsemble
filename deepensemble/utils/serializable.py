import sys

from six.moves import cPickle

__all__ = ['Serializable']


class Serializable(object):
    """ Class for save an load data in a object.
    """

    def __init__(self, data=None):
        super(Serializable, self).__init__()
        self.__data = data

    def get_data(self):
        return self.__data

    def load(self, filename):
        """ Load model from file.

        Parameters
        ----------
        filename : str
            Path of file where recovery data of model.
        """
        file_model = open(filename, 'rb')
        tmp_dict = cPickle.load(file_model)
        file_model.close()
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """ Save data to file.

        Parameters
        ----------
        filename : str
            Path of file where storage data of model.
        """
        sys.setrecursionlimit(10 ** 4)
        file_model = open(filename, 'wb')
        cPickle.dump(self.__dict__, file_model, protocol=cPickle.HIGHEST_PROTOCOL)
        file_model.close()

    def get_serializable(self, data):
        self.__data = data
        return self
