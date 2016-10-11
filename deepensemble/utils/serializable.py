import pickle
import sys


__all__ = ['Serializable']


class Serializable(object):

    def load(self, filename):
        """ Load model from file.

        Parameters
        ----------
        filename : str
            Path of file where recovery data of model.
        """
        file_model = open(filename, 'rb')
        tmp_dict = pickle.load(file_model)
        file_model.close()
        self.__dict__.update(tmp_dict)

    def save(self, filename):
        """ Save data to file.

        Parameters
        ----------
        filename : str
            Path of file where storage data of model.
        """
        sys.setrecursionlimit(10**4)
        file_model = open(filename, 'wb')
        pickle.dump(self.__dict__, file_model, -1)
        file_model.close()