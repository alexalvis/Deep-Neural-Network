import numpy as np
#import matplotlib.pyplot as plt
import scipy.optimize  # For check_grad, approx_fprime

class RNN:
    def __init__ (self, numHidden, numInput, numOutput):
        self.numHidden = numHidden
        self.numInput = numInput
        self.numOutput = numOutput
        self.U = np.random.randn(numHidden, numHidden) * 1e-1
        self.V = np.random.randn(numHidden, numInput) * 1e-1
        self.w = np.random.randn(numHidden) * 1e-1
        # TODO: IMPLEMENT ME

    def backward (self, x, y):
        # TODO: IMPLEMENT ME
        # size y [50,]
        h, r_yhat = self.forward(x)
        yhat = r_yhat.reshape(50,)
        # print len(y)
        # print yhat.shape
        gradient_w = np.zeros(self.numHidden)
        gradient_U = np.zeros((self.numHidden,self.numHidden))
        gradient_V = np.zeros((self.numHidden,self.numInput))
        for t in np.arange(len(x))[::-1]:
            gradient_w += (yhat[t] - y[t]) * h[t]
            a = self.w.T.dot(yhat[t]-y[t]) * (1-np.power(h[t],2))
            for k in np.arange(t + 1)[::-1]:
                gradient_U = gradient_U + np.outer(a, h[k-1])
                gradient_V = gradient_V + np.outer(a, x[k])
                a = self.U.T.dot(a) * (1- np.power(h[k-1],2))
        return gradient_U, gradient_V, gradient_w

    def forward (self, x):
        # size of x [50,1]
        H = np.zeros((len(x)+1, self.numHidden))
        yhat = np.zeros((len(x),numInput))
        for t in np.arange(len(x)):
            temp = self.V.dot(x[t])
            H[t] =  np.tanh(self.U.dot(H[t-1].T) + temp.reshape(6,))
            yhat[t] = self.w.dot(H[t])
        #yhat = H[0:-1].dot(self.w)
        return H, yhat

    def J(self, x, y):
        h, r_yhat = self.forward(x)
        yhat = r_yhat.reshape(50,)
        J = (yhat - y) ** 2
        return np.sum(J)

    def nextBatch(self, x, y):
        # total_size = x.shape[0]
        # for i in np.arange(0,total_size,1):
        #     yield (x[i:i + 1], y[i:i + 1])
        yield (x, y)

    def SGD_J(self, x, y, lr=0.01, n_epoch=200):
        start = 0;
        while (start < n_epoch):
            for x, y in self.nextBatch(x, y):
                grad1, grad2 ,grad3= self.backward(x, y)
                self.U -= lr * grad1
                self.V -= lr * grad2
                self.w -= lr * grad3
            start = start + 1
            print "Epoch:", '%04d' % (start)
            print "Tr Loss with J: %.6f" % self.J(x, y)

# From https://medium.com/@erikhallstrm/hello-world-rnn-83cd7105b767
def generateData ():
    total_series_length = 50
    echo_step = 2  # 2-back task
    batch_size = 1
    x = np.random.choice(2, total_series_length, p=[0.5, 0.5])
    y = np.roll(x, echo_step)
    y[0:echo_step] = 0
    y = list(y)
    return (x, y)

if __name__ == "__main__":
    xs, ys = generateData()
    print xs
    print ys
    numHidden = 6
    numInput = 1
    numTimesteps = len(xs)
    rnn = RNN(numHidden, numInput, 1)
    rnn.SGD_J(xs,ys)
    print "Final Loss: %.6f" % rnn.J(xs,ys)
    # TODO: IMPLEMENT ME
