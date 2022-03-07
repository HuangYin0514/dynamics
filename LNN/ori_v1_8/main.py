import datetime
import random

import numpy as np
import torch

from dataset import DoublePendulumData, DataGenerator
from trainer import Trainer


def init_random_state():
    random_seed = 3404
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.cuda.manual_seed(random_seed)
    np.random.seed(random_seed)  # Numpy module.
    random.seed(random_seed)  # Python random module.
    torch.backends.cudnn.deterministic = True  # speed up computation
    torch.backends.cudnn.benchmark = True


init_random_state()

if __name__ == '__main__':
    startTime = datetime.datetime.now()

    nIter = 300

    if not torch.cuda.is_available():
        nIter = 300

        print("use cpu!!!")

    doublePendulumData = DoublePendulumData()

    x, y = doublePendulumData.x, doublePendulumData.y

    doublePendulum_dataset = DataGenerator(x, y)

    train_loader = torch.utils.data.DataLoader(
        doublePendulum_dataset,
        batch_size=3000,
    )

    trainer = Trainer()
    trainer.train(train_loader, nIter=nIter)

    torch.save(trainer.model, "result/model_final.pkl")

    endTime = datetime.datetime.now()
    print("produce: " + str(endTime - startTime))

    print("done")
