import numpy as np
import scipy.io
from torch.utils import data

from utils import get_device, to_tensor


class BurgerData():
    '''
        Data for learning the antiderivative operator.
    '''

    def __init__(self):
        super().__init__()

        path = 'data/Burger.mat'  # Please use the matlab script to generate data
        data = scipy.io.loadmat(path)
        usol = np.array(data['output'])

        N = usol.shape[0]  # number of total input samples

        self.N_train = 1  # number of input samples used for training

        self.P_ics_train = 101  # number of locations for evulating the initial condition
        self.P_bcs_train = 100  # number of locations for evulating the boundary condition
        self.P_res_train = 2500  # number of locations for evulating the PDE residual

        self.u0_train = usol[:self.N_train, 0, :]  # input samples

        self.__init_data()

    def __init_data(self):
        ics_train = list(map(self.generate_one_ics_training_data, self.u0_train))
        self.u_ics_train = np.array(list(map(lambda x: x[0], ics_train))).reshape(self.N_train * self.P_ics_train, -1)
        self.y_ics_train = np.array(list(map(lambda x: x[1], ics_train))).reshape(self.N_train * self.P_ics_train, -1)
        self.s_ics_train = np.array(list(map(lambda x: x[2], ics_train))).reshape(self.N_train * self.P_ics_train, -1)

        bcs_train = list(map(self.generate_one_bcs_training_data, self.u0_train))
        self.u_bcs_train = np.array(list(map(lambda x: x[0], bcs_train))).reshape(self.N_train * self.P_bcs_train, -1)
        self.y_bcs_train = np.array(list(map(lambda x: x[1], bcs_train))).reshape(self.N_train * self.P_bcs_train, -1)
        self.s_bcs_train = np.array(list(map(lambda x: x[2], bcs_train))).reshape(self.N_train * self.P_bcs_train, -1)

        res_train = list(map(self.generate_one_res_training_data, self.u0_train))
        self.u_res_train = np.array(list(map(lambda x: x[0], res_train))).reshape(self.N_train * self.P_res_train, -1)
        self.y_res_train = np.array(list(map(lambda x: x[1], res_train))).reshape(self.N_train * self.P_res_train, -1)
        self.s_res_train = np.array(list(map(lambda x: x[2], res_train))).reshape(self.N_train * self.P_res_train, -1)

    # Geneate ics training data corresponding to one input sample
    def generate_one_ics_training_data(self, u0, P=101):
        t_0 = np.zeros((P, 1))
        x_0 = np.linspace(0, 1, P)[:, None]

        y = np.hstack([t_0, x_0])
        u = np.tile(u0, (P, 1))
        s = u0

        return u, y, s

    # Geneate bcs training data corresponding to one input sample
    def generate_one_bcs_training_data(self, u0, P=100):
        t_bc = np.random.uniform(size=(P, 1))
        x_bc1 = np.zeros((P, 1))
        x_bc2 = np.ones((P, 1))

        y1 = np.hstack([t_bc, x_bc1])  # shape = (P, 2)
        y2 = np.hstack([t_bc, x_bc2])  # shape = (P, 2)

        u = np.tile(u0, (P, 1))
        y = np.hstack([y1, y2])  # shape = (P, 4)
        s = np.zeros((P, 1))

        return u, y, s

    # Geneate res training data corresponding to one input sample
    def generate_one_res_training_data(self, u0, P=2500):
        t_res = np.random.uniform(size=(P, 1))
        x_res = np.random.uniform(size=(P, 1))

        u = np.tile(u0, (P, 1))
        y = np.hstack([t_res, x_res])
        s = np.zeros((P, 1))

        return u, y, s

    # Data generator


class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, batch_size):
        'Initialization'

        self.u = u
        self.y = y
        self.s = s

        self.N = self.s.shape[0]  # all dataset number
        self.batch_size = self.N

        self.device = get_device()

    def __getitem__(self, index):
        'Generate one batch of data'
        inputs, outputs = self.__data_generation()
        return inputs, outputs

    def __data_generation(self):
        'Generates data containing batch_size samples'
        idx = np.random.choice(self.N, (self.batch_size,), replace=False)  # False: 同一个元素只能被选取一次。
        u = self.u[idx, :]
        y = self.y[idx, :]
        s = self.s[idx, :]

        # Construct batch
        inputs = (to_tensor(u), to_tensor(y))
        outputs = to_tensor(s)

        return inputs, outputs


if __name__ == '__main__':
    burgerData = BurgerData()

    batch_size = 50000
    u_ics_train, y_ics_train, s_ics_train = burgerData.u_ics_train, burgerData.y_ics_train, burgerData.s_ics_train
    u_bcs_train, y_bcs_train, s_bcs_train = burgerData.u_bcs_train, burgerData.y_bcs_train, burgerData.s_bcs_train
    u_res_train, y_res_train, s_res_train = burgerData.u_res_train, burgerData.y_res_train, burgerData.s_res_train

    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

    outputs = next(iter(ics_dataset))
