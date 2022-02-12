import random

import numpy as np
import torch
import torch.backends.cudnn

from dataset import IntegralData, DataGenerator
from trainer import Trainer
from utils import get_device

device = get_device()


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

    data = IntegralData()
    X_train, y_train = data.X_train, data.y_train
    train_dataset = DataGenerator(u=X_train[0], y=X_train[1], s=y_train)

    trainer = Trainer()
    trainer.train(train_dataset, nIter=nIter)

    torch.save(trainer.model, "result/model_final.pkl")

    print("done")
