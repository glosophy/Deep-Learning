import numpy as np
import matplotlib.pyplot as plt

S1 = 5  # neurons
R = 1  # dimensions
p = np.linspace(-2, 2, 100)  # p between (-2,2)
alpha = 0.02  # learning rate

#Randomly initialize Weights and Biases
np.random.seed(4)
w1 = np.random.uniform(-0.5, 0.5, (S1, R))
b1 = np.random.uniform(-0.5, 0.5, (S1, 1))
w2 = np.random.uniform(-0.5, 0.5, (1, S1))
b2 = np.random.uniform(-0.5, 0.5, (1, 1))


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


# Update weights and biases
def update(s1, s2, w1, w2, b1, b2, alpha, a1, p):
    w2_new = w2 - alpha * np.dot(s2, a1.T)
    w1_new = w1 - alpha * np.dot(s1, p)
    b2_new = b2 - alpha * s2
    b1_new = b1 - alpha * s1
    return w2_new, w1_new, b2_new, b1_new


# Define neural network function
def NNF(w1, b1, w2, b2, alpha, t, epochs = 15000):
    MSE = np.zeros(epochs)

    for i in range(epochs):
        e = np.zeros(len(p))

        for j in range(len(p)):
            a1, n1, a2, n2 = forward_propagation(p[j], w1, b1, w2, b2)
            e[j] = t[j] - a2.reshape(-1).item()
            s1, s2 = backward(a1, w2, e[j])
            w2, w1, b2, b1 = update(s1, s2, w1, w2, b1, b2, alpha, a1, p[j])
        MSE[i] = np.dot(e.T, e)

        print(MSE[i])
    return e, MSE, w1, w2, b2, b1


# Call NNF
e, MSE, w1, w2, b2, b1 = NNF(w1, b1, w2, b2, alpha, t)

#Plot
q = np.zeros((len(p)))
for i in range(len(p)):
    _, _, q[i], _ = forward_propagation(p[i], w1, b1, w2, b2)

plt.figure(1)
plt.plot(p, t, label='Original function', color = 'green')
plt.plot(p, q, label='Network function', color = 'blue')
plt.legend()
plt.show()

#Plot squared error for epochs
plt.figure(2)
plt.loglog(range(len(MSE)), MSE)
plt.show()
