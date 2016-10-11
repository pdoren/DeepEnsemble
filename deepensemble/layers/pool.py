from .layer import Layer

__all__ = ['MaxPool1D', 'MaxPool2D', 'Pool1D', 'Pool2D']


# noinspection PyUnusedLocal
class PoolBase(Layer):
    def __init__(self, pool_size, stride=None, pad=0, ignore_border=True, mode='max'):
        super(PoolBase, self).__init__(input_shape=None, output_shape=None, non_linearity=None, exclude_params=True)

    def output(self, x):
        return x


class MaxPool1D(PoolBase):
    def __init__(self, pool_size):
        super(MaxPool1D, self).__init__(pool_size=pool_size, mode='max')


class MaxPool2D(PoolBase):
    def __init__(self, pool_size):
        super(MaxPool2D, self).__init__(pool_size=pool_size, mode='max')


class Pool1D(PoolBase):
    def __init__(self, pool_size):
        super(Pool1D, self).__init__(pool_size=pool_size)


class Pool2D(PoolBase):
    def __init__(self, pool_size):
        super(Pool2D, self).__init__(pool_size=pool_size)
