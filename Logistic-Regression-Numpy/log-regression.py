import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import cv2, os, pathlib
from sklearn import preprocessing
import collections


cwd = os.getcwd()
dir = cwd + '/images/'
data_dir = pathlib.Path(dir)

print(cwd)
print(data_dir)

# Read images
print("Starting images and label pre-processing...")

label_data = []
images_data = []
for subdir, dirs, files in os.walk(dir):
    for file in files:
        image = os.path.join(subdir, file)
        img = cv2.imread(image)
        # handle non-readable images
        if img is None:
            pass
        else:
            images_data.append(img)
            # read directory and append labels
            label = (subdir.split('images/')[1])
            label_data.append(label)

print("Images and labels successfully pre-processed!")

# look at labels and images shape
labels_data = np.array(label_data)
print("Labels shape:", labels_data.shape)
images_data = np.array(images_data)
print("Images shape:", images_data.shape)

# one-hot encoding: Convert text-based labels to int
le = preprocessing.LabelEncoder()
le.fit(labels_data)

# confirm we have 2 unique classes
print('Unique classes:',le.classes_)
integer_labels = le.transform(labels_data)

# count images
image_count = len(list(images_data))
print("Number of images:", image_count)
print("")

# count number of images in every category
print("Number of images in each class:")
print(collections.Counter(labels_data).keys())
print(collections.Counter(labels_data).values())


# split train/test
x_train, x_test, y_train, y_test = train_test_split(images_data, integer_labels, random_state = 42, test_size = 0.2, stratify = integer_labels)

# Example of a picture
index = 507
plt.imshow(x_train[index])
plt.show()

if y_train[index] == 1:
  print ("y = " + str(y_train[index]) + ", it's a lake picture.")
else:
  print ("y = " + str(y_train[index]) + ", it's an airplane picture.")

m_train = len(x_train)
m_test = len(x_test)
num_px = x_train.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("x_train shape: " + str(x_train.shape))
print ("y_train shape: " + str(y_train.shape))
print ("x_test shape: " + str(x_test.shape))
print ("y_test shape: " + str(y_test.shape))

# Reshape the training and test examples
x_train_flatten = x_train.reshape(x_train.shape[0], -1).T
x_test_flatten = x_test.reshape(x_test.shape[0], -1).T

print ("x_train_flatten shape: " + str(x_train_flatten.shape))
print ("y_train shape: " + str(y_train.shape))
print ("x_test_flatten shape: " + str(x_test_flatten.shape))
print ("y_test shape: " + str(y_test.shape))

# standarize images
X_train = x_train_flatten/255.
X_test = x_test_flatten/255.


# HELPER FUNCTION: sigmoid
def sigmoid(z):
    """
    Compute the sigmoid of z

    Arguments:
    z -- A scalar or numpy array of any size.

    Returns:
    s -- sigmoid(z)
    """

    s = 1 / (1 + np.exp(-z))

    return s


# INITIALIZE PARAMETERS
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.

    Argument:
    dim -- size of the w vector we want (or number of parameters in this case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    s = (dim, 1)
    w = np.zeros(s)
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))

    return w, b


# FORWARD AND BACKWARD PROPAGATION FUNCTION
def propagate(w, b, X, Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Returns:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute activation

    cost = (-1 / m) * (np.sum((Y * (np.log(A))) + ((1 - Y) * (np.log((1 - A)+0.00000000001)))))


    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1 / m) * (np.dot(X, ((A - Y).T)))
    db = (np.sum(A - Y)) * (1 / m)

    assert (dw.shape == w.shape)
    assert (db.dtype == float)
    cost = np.squeeze(cost)
    assert (cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


# OPTIMIZATION FUNCTION
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    """
    This function optimizes w and b by running a gradient descent algorithm

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if lake, 1 if airplane), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """

    costs = []

    for i in range(num_iterations):

        # Cost and gradient calculation
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update rule
        w = w - (learning_rate * dw)
        b = b - (learning_rate * db)

        # Record the costs
        if i % 100 == 0:
            costs.append(cost)

        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


# GRADED FUNCTION: predict
def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of an airplane being present in the picture
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] <= 0.5:
            Y_prediction[0, i] += 0
        else:
            Y_prediction[0, i] += 1

    assert (Y_prediction.shape == (1, m))

    return Y_prediction


# MODEL FUNCTION:
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost= True):
    """
    Builds the logistic regression model by calling the function you've implemented previously

    Arguments:
    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations

    Returns:
    d -- dictionary containing information about the model.
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # Gradient descent
    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost=True)

    # Retrieve parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # Predict test/train set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d


d = model(X_train, y_train, X_test, y_test, num_iterations = 2000, learning_rate = 0.005, print_cost = True)


# Plot learning curve (with costs)
costs1 = np.squeeze(d['costs'])
plt.plot(costs1)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()



learning_rates = [0.01, 0.001, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(X_train, y_train, X_test, y_test, num_iterations = 1500, learning_rate = i, print_cost = False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')

plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()
