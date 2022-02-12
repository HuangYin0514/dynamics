import numpy as np
import torch
from matplotlib import pyplot as plt

from trainer import Trainer
from utils import get_device
import matplotlib.image as mpimg

import  cv2
device = get_device()



if __name__ == '__main__':
    img = mpimg.imread("result/deeponet.png")
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

