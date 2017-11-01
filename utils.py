import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import random


class SimpleDataIterator():
    def __init__(self, data):
        self.data = data
        self.size = np.size(data,0)
        self.epochs = 0
        self.shuffle()

    def shuffle(self):
        np.random.shuffle(self.data)  # fatal error random.shuffle(numpy array) has some error, use np.random.shuffle, it select part its elements, so some elements become more and more
        self.cursor = 0

    def next_batch(self, n):
        if self.cursor+n-1 > self.size:
            self.epochs += 1
            self.shuffle()
        res = self.data[self.cursor:self.cursor+n]
        self.cursor += n
        return res

def plot_all_complex(data,num_figure,it,L, DATA, LOSS):
 
    "Plot the six MNIST images separately."
    f, axarr = plt.subplots(num_figure, num_figure)
    for i in range(num_figure):
        for j in range(num_figure):
            axarr[i,j].matshow(np.reshape(data[i*num_figure+j],[28,28]), cmap = matplotlib.cm.binary)
            axarr[i,j].axis('off')
            
    plt.suptitle('Plot of  '+str(it), fontsize=25)

    plt.savefig('out/{}_{}_{}_{}.png'
                    .format(DATA,LOSS,L,it), bbox_inches='tight')
    plt.close()