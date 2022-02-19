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

        self.N_train = 9  # number of input samples used for training

        self.P_ics_train = 101  # number of locations for evulating the initial condition
        self.P_bcs_train = 100  # number of locations for evulating the boundary condition
        self.P_res_train = 2500  # number of locations for evulating the PDE residual

        self.u0_train = usol[:self.N_train, 0, :]  # input samples

        self.__init_data()

    def __init_data(self):
        def get_all_training_data(generate_method, P, u0_train=self.u0_train, N_train=self.N_train):
            u, y, s = [], [], []
            for u0 in u0_train:
                u_temp, y_temp, s_temp = generate_method(u0)
                u.append(u_temp)
                y.append(y_temp)
                s.append(s_temp)
            u = np.array(u).reshape(N_train * P, -1)
            y = np.array(y).reshape(N_train * P, -1)
            s = np.array(s).reshape(N_train * P, -1)
            return u, y, s

        self.u_ics_train, self.y_ics_train, self.s_ics_train = get_all_training_data(
            self.generate_one_ics_training_data, self.P_ics_train)
        self.u_bcs_train, self.y_bcs_train, self.s_bcs_train = get_all_training_data(
            self.generate_one_bcs_training_data, self.P_bcs_train)
        self.u_res_train, self.y_res_train, self.s_res_train = get_all_training_data(
            self.generate_one_res_training_data, self.P_res_train)

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
    def __init__(self, u, y, s):
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

    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train)
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train)
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train)

    outputs = next(iter(ics_dataset))
