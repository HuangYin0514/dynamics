import random

import numpy as np
import torch
import torch.backends.cudnn

from dataset import DataGenerator, BurgerData
from trainer import Trainer


def init_random_state():
    random_seed = 3407
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # speed up compution
    torch.backends.cudnn.benchmark = True


init_random_state()

if __name__ == '__main__':

    nIter = 500

    if not torch.cuda.is_available():
        nIter = 10
        print("use cpu!!!")

    burgerData = BurgerData()

    x_ibcs_train, u_ibcs_train, s_ibcs_train = burgerData.x_ibcs_train, burgerData.u_ibcs_train, burgerData.s_ibcs_train
    x_res_train, u_res_train, s_res_train = burgerData.x_res_train, burgerData.u_res_train, burgerData.s_res_train

    ibcs_dataset = DataGenerator(u_ibcs_train, x_ibcs_train, s_ibcs_train)
    res_dataset = DataGenerator(u_res_train, x_res_train, s_res_train)

    trainer = Trainer()
    trainer.train(ibcs_dataset, res_dataset, nIter=nIter)

    torch.save(trainer.model, "result/model_final.pkl")

    print("done")
