import numpy as np
import matplotlib.pyplot as plt

S1 = 5  # neurons
R = 1  # dimensions
p = np.linspace(-2, 2, 100)  # p between (-2,2)
alpha = 0.02  # learning rate


#Initialize weights and biases
np.random.seed(5)
w1 = np.random.uniform(-0.5, 0.5,(S1, R))
b1 = np.random.uniform(-0.5, 0.5,(S1, 1))
w2 = np.random.uniform(-0.5, 0.5,(1, S1))
b2 = np.random.uniform(-0.5, 0.5,(1, 1))


# Define approximation function
def g(n):
    app = np.exp(-np.abs(n)) * np.sin(np.pi * n)
    return app


# Define activation function
def logsig(n):
    a = 1 / (1 + np.exp(-n))
    return a


# Define target function
t = g(p)


 # Define forward propagation function
def forward_propagation(p, w1, b1, w2, b2):
    n1 = np.dot(w1, p) + b1
    a1 = logsig(n1)

    n2 = np.dot(w2, a1) + b2
    a2 = n2
    return a1, n1, a2, n2


# Define backpropagation to calculate sensitivities
def backward(a1, w2, e):
    s2 = (-2 * 1) * e
    s1 = np.dot(np.diag(((1 - a1) * a1).flatten()), w2.T) * s2
    return s1, s2


def NNF_Batch(p, w1, b1, w2, b2, alpha, t, epochs = 35000):
    MSE_batch = np.zeros(epochs)
    for j in range(epochs):
        e = np.zeros(len(p))
        s1_sum = 0
        s2_sum = 0
        r2 = 0
        r1 = 0
        for i in range(len(p)):
            a1, n1, a2, n2 = forward_propagation(p[i], w1, b1, w2, b2)
            e[i] = t[i] - a2.reshape(-1).item()
            s1, s2 = backward(a1, w2, e[i])
            s1_sum += s1
            s2_sum += s2
            r2 += np.dot(s2, a1.T)
            r1 += np.dot(s1, p[i])
        w2 = w2 - alpha * r2
        w1 = w1 - alpha * r1
        b2 = b2 - alpha * s2_sum
        b1 = b1 - alpha * s1_sum
        MSE_batch[j] = np.dot(e.T, e)
        print(MSE_batch[j])
    return e, MSE_batch, w1, w2, b2, b1


e, MSE_batch, w1, w2, b2, b1 = NNF_Batch(p, w1, b1, w2, b2, alpha, t)


#Plot original vs. network
q = np.zeros((len(p)))
for i in range(len(p)):
    _, _, q[i], _ = forward_propagation(p[i], w1, b1, w2, b2)

plt.figure(1)
plt.plot(p, t, label = 'Original function', color = 'green')
plt.plot(p, q, label = 'Network function', color = 'blue')
plt.legend()
plt.show()


#Plot squared error for epochs
plt.figure(2)
plt.plot(range(len(MSE_batch)), MSE_batch)
plt.show()
