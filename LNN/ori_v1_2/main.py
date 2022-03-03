import datetime

import torch

from dataset import DoublePendulumData, DataGenerator
from trainer import Trainer

if __name__ == '__main__':
    startTime = datetime.datetime.now()

    nIter = 5000

    if not torch.cuda.is_available():
        nIter = 5
        print("use cpu!!!")

    doublePendulumData = DoublePendulumData()

    x, y = doublePendulumData.x, doublePendulumData.y

    doublePendulum_dataset = DataGenerator(x, y)

    trainer = Trainer()
    trainer.train(doublePendulum_dataset, nIter=nIter)

    torch.save(trainer.model, "result/model_final.pkl")

    endTime = datetime.datetime.now()
    print("produce: " + str(endTime - startTime))

    print("done")
