import matplotlib.image as mpimg
from matplotlib import pyplot as plt

if __name__ == '__main__':
    img = mpimg.imread("result/deeponet.png")
    plt.figure(figsize=(15,8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
