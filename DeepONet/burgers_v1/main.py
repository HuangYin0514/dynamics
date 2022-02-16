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

    nIter = 5000

    if not torch.cuda.is_available():
        nIter = 10
        print("use cpu!!!")

    burgerData = BurgerData()

    batch_size = 50000
    u_ics_train, y_ics_train, s_ics_train = burgerData.u_ics_train, burgerData.y_ics_train, burgerData.s_ics_train
    u_bcs_train, y_bcs_train, s_bcs_train = burgerData.u_bcs_train, burgerData.y_bcs_train, burgerData.s_bcs_train
    u_res_train, y_res_train, s_res_train = burgerData.u_res_train, burgerData.y_res_train, burgerData.s_res_train

    ics_dataset = DataGenerator(u_ics_train, y_ics_train, s_ics_train, batch_size)
    bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
    res_dataset = DataGenerator(u_res_train, y_res_train, s_res_train, batch_size)

    trainer = Trainer()
    trainer.train(ics_dataset, bcs_dataset, res_dataset, nIter=nIter)

    torch.save(trainer.model, "result/model_final.pkl")

    print("done")
