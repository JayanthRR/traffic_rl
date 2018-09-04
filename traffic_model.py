import numpy as np
# import matplotlib.pyplot as plt
import scipy


def exponential(lamda, k):
    return np.random.exponential(scale=1/lamda, size=k)


def gaussian(mean, sigma, k):
    return np.random.normal(loc=mean, scale=sigma, size=k)


def generate_column_stochastic_matrix(n):
    # Doesn't generate a sparse matrix. The degree of each link can be as high as n-1.
    # Todo: Think of how to make a sparse matrix (to mimic road network geometry)
    A = np.random.rand(n,n)
    temp = np.random.choice([0,1], (n,n), p=[0.9,0.1])
    A = np.multiply(A, temp)
    A = A/A.sum(axis=0)
    return A


def generate_signal(st):
    # deterministically right shifts each signal in the signal vector
    s = []
    s.append(st[-1])
    s.extend(st[:-1])
    return s


def signalized(A, signal):

    # signal is a n length vector, which determines the signal on that specific link. Takes values 0 or 1
    # convention: 1 for red, 0 for green
    B = np.zeros_like(A)
    for i in range(len(signal)):
        if signal[i] == 1:
            B[:, i] = [1 if j == i else 0 for j in range(len(signal))]
        else:
            B[:, i] = A[:, i]

    return B

# def quantized_state(xt, size=5, type_='uniform'):
#     if type_=='uniform':


def state_transition(A, xt, wt, mean=0, sigma=0.1):
    c = 0.1
    if wt is None:
        wt = 1 * gaussian(mean, sigma, len(xt))

    x_t1 = np.matmul(A, xt) + wt

    x_t1 = np.minimum(np.maximum(x_t1, np.zeros(len(xt))), c * np.ones(len(xt)))

    return x_t1


if __name__ == "__main__":
    time_steps = 100
    size = 15
    X = []
    S = []
    xt = x_init = np.ones(size) / size
    st = np.random.choice([0, 1], size=size, p=[1. / 3, 2. / 3])
    mean = 0
    X.append(x_init)
    S.append(st)
    A = generate_column_stochastic_matrix(size)
    # print(A, A.sum(axis=0))

    for i in range(time_steps):
        xt, st = state_transition(A, xt, wt=None, mean=mean, sigma=0.1)

        X.append(xt)
        S.append(st)

    X = np.array(X)
    S = np.array(S)
    for i in range(size):
        plt.plot(X[:, i])

    plt.show()

