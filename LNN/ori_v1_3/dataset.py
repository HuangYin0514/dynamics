import numpy as np
from torch.utils import data

from physics import analytical_fn
from utils import get_device, to_tensor


class DoublePendulumData():
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()

        self.batch_size = 512
        self.minibatch_per = 2000

        self.vfnc = analytical_fn

        self.__init_data()

    def __init_data(self):
        self.x, self.y = self.get_derivative_dataset()

    def get_derivative_dataset(self):
        y0 = np.concatenate([np.random.uniform(size=(self.batch_size * self.minibatch_per, 2)) * 2 * np.pi,
                             (np.random.uniform(size=(self.batch_size * self.minibatch_per, 2)) -0.5)*10*2],
                            axis=1)
        return y0, np.array(list(map(lambda x: self.vfnc(x), y0)))

        # y0 = np.array([[2.25210454, 4.56980437, 0.01023673, 0.08267299]])
        # return y0, np.array(list(map(lambda x: self.vfnc(x), y0)))



class DataGenerator(data.Dataset):
    def __init__(self, x, y):
        'Initialization'

        self.x = x
        self.y = y

        self.N = self.x.shape[0]  # all dataset number
        self.batch_size = self.N

        self.device = get_device()

    def __getitem__(self, index):
        'Generate one batch of data'
        x, y = self.__data_generation(index)
        return x, y

    def __data_generation(self, index):
        'Generates data containing batch_size samples'
        minibatch=2000
        x = self.x[index*minibatch:(index+1)*minibatch]
        y = self.y[index*minibatch:(index+1)*minibatch]

        # Construct batch
        inputs = to_tensor(x)
        outputs = to_tensor(y)

        return inputs, outputs


if __name__ == '__main__':
    doublePendulumData = DoublePendulumData()

    x, y = doublePendulumData.x, doublePendulumData.y

    doublePendulum_dataset = DataGenerator(x, y)

    outputs = next(iter(doublePendulum_dataset))
