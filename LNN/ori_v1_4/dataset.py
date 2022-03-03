import numpy as np
import torch
from torch.utils import data
from torch.utils.data import DataLoader

from physics import analytical_fn
from utils import get_device, to_tensor

class DoublePendulumData():
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()

        self.batch_size = 256
        self.minibatch_per = 500

        self.vfnc = analytical_fn

        self.__init_data()

    def __init_data(self):
        self.x, self.y = self.get_derivative_dataset()

    def get_derivative_dataset(self):
        y0 = np.concatenate([np.random.uniform(size=(self.batch_size * self.minibatch_per, 2)) * 2 * np.pi,
                             (np.random.uniform(size=(self.batch_size * self.minibatch_per, 2)) * 0.1)],
                            axis=1)
        return y0, np.array(list(map(lambda x: self.vfnc(x), y0)))


class DataGenerator(data.Dataset):
    def __init__(self, x, y):
        'Initialization'

        self.x = x
        self.y = y

        self.N = self.x.shape[0]  # all dataset number
        self.batch_size = self.N

        self.device = get_device()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        'Generate one batch of data'
        x, y = self.__data_generation(index)
        return x, y

    def __data_generation(self,index):
        'Generates data containing batch_size samples'

        x = self.x[index]
        y = self.y[index]

        # Construct batch
        inputs = to_tensor(x)
        outputs = to_tensor(y)

        return inputs, outputs


if __name__ == '__main__':
    doublePendulumData = DoublePendulumData()

    x, y = doublePendulumData.x, doublePendulumData.y

    doublePendulum_dataset = DataGenerator(x, y)

    outputs = next(iter(doublePendulum_dataset))

    train_loader = torch.utils.data.DataLoader(
        doublePendulum_dataset,
        batch_size=2000,
    )

    for i in train_loader:
        print(i)
        break
