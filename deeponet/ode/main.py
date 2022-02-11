from deeponet.ode.dataset import IntegralData, DataGenerator
from trainer import Trainer

if __name__ == '__main__':
    data = IntegralData()
    X_train, y_train = data.X_train, data.y_train
    train_dataset = DataGenerator(u=X_train[0], y=X_train[1], s=y_train)

    trainer = Trainer()
    trainer.train(train_dataset, nIter=1)

    print("done")
