import numpy as np
import cv2, collections
import os.path
import zipfile
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D
from keras.models import Model
from keras.utils import *
import keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt


cwd = os.getcwd()
print('Current working directory:', cwd)
print('')
parent = os.path.abspath(os.path.join(cwd, os.pardir))
print('Parent directory is:', parent)

#Extract zip into folder
with zipfile.ZipFile(parent + '/Logistic-Regression-Numpy/images.zip', 'r') as zip_ref:
    zip_ref.extractall(cwd)

# Define new path to work on the images
dir = cwd + '/images/'

#::-------------------------------------
# Data-pre processing
#::-------------------------------------

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

#::-------------------------------------
# Shape checking
#::-------------------------------------

# Normalize image vectors
X_train = x_train/255.
X_test = x_test/255.

# Reshape
Y_train = y_train.T
Y_test = y_test.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

#::-------------------------------------
# Build keras model
#::-------------------------------------

def kerasModel(input_shape):
    """
    Implementation of the kerasModel.

    Arguments:
    input_shape -- shape of the images of the dataset
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    # Zero-Padding: pads the border of X_input with zeroes
    X = ZeroPadding2D((3, 3))(X_input)

    # CONV -> BN -> RELU Block applied to X
    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)
    X = BatchNormalization(axis=3, name='bn0')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool0')(X)

    X = Conv2D(32, (3, 3), strides=(1, 1), name='conv1')(X)
    X = BatchNormalization(axis=3, name='bn1')(X)
    X = Activation('relu')(X)

    # MAXPOOL
    X = MaxPooling2D((2, 2), name='max_pool1')(X)

    # FLATTEN X (convert it to a vector) + FULLYCONNECTED
    X = Flatten()(X)
    X = Dense(1, activation='sigmoid', name='fc')(X)

    # Create model. This creates your Keras model instance
    model = Model(inputs=X_input, outputs=X, name='kerasModel')

    return model

#::-------------------------------------
# Create / Compile / Train / Evaluate
#::-------------------------------------

# Step 1: create the model
kerasModel = kerasModel(X_train.shape[1:])

# Step 2: compile the model
kerasModel.compile('adam', 'binary_crossentropy', metrics=['accuracy'])

# Step 3: train the model
kerasModel.fit(x = X_train, y = Y_train, batch_size = 32, epochs = 40)

# Step 4: evaluate the model
preds = kerasModel.evaluate(x = X_test, y = Y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))


# Print the details of the layers in a table with the sizes of its inputs/outputs
kerasModel.summary()

# Plot the graph in a nice layout
plot_model(kerasModel, to_file='kerasModel.png')
# SVG(model_to_dot(kerasModel).create(prog='dot', format='svg'))
