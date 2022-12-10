from kernel import *

class SVM:
    kernel: Kernel
    phi:callable
    _w:np.ndarray


    @property
    def w(self):
        return self._w
    @w.setter
    def w(self, w):
        self._w = w



if __name__ == '__main__':
    x1 = np.array([1, 1, 1])
    x2 = np.array([1, 1, 1])

    K = Kernel("poly", d=3)
    print(K(x1, x2))