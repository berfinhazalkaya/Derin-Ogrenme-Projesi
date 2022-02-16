import cv2
import numpy as np
import os
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

# We will use the pictures in the folder named "train" in the training.
trainFolder = 'train'

# We will use the pictures in the folder named "test" in the test.
testFolder = 'test'

# we will resize the images used for training and testing to the same size (50x50 pixels).
pictureSize = 50

# let's define the learning rate as 0.001.
learningRate = 1e-3

# Let's give a name to the model we will create.
modelName = 'cat-dog-prediction'

### OBTAINING LABEL INFORMATION FROM FILE NAMES ###

#Let's define a function called #create_tag.
# in filenames with this function
#We will detect the "cat" or "dog" tags in # a.
# function outputs [1 0] if filename is "cat", [0 1] if "dog".

def GenerateLabel(pictureName):
    objectType = pictureName.split('.')[-3] # get "cat" or "dog" in filename
    if objectType == 'cat':
        return np.array([1, 0])
    elif objectType == 'dog':
        return np.array([0, 1])


### CONVERTING PICTURES TO MATRIX ###

# Generate training data from the images in the train folder that can be used in training.
# The generated training data is written to the file named "training_Data.npy".
# Shuffle the data in the function.
# images are read as gray and resized to 50x50 pixels.

def GenerateTrainingData():
    generatedTrainingData = []
    for img in tqdm(os.listdir(trainFolder)):
        filePath = os.path.join(trainFolder, img)
        pictureData = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        pictureData = cv2.resize(pictureData, (pictureSize, pictureSize))
        generatedTrainingData.append([np.array(pictureData), GenerateLabel(img)])
    shuffle(generatedTrainingData)
    np.save('train_Data.npy', generatedTrainingData)
    return generatedTrainingData


# Generate test data from the pictures in the test folder so that it can be used in training.
# the generated test data is written to the file named "test_Data.npy"
# Shuffle the data in the function.
# images are read as gray and resized to 50x50 pixels.

def GenerateTestData():
    generatedTestData = []
    for img in tqdm(os.listdir(testFolder)):
        filePath = os.path.join(testFolder, img)
        pictureNumber = img.split('.')[0]
        pictureData = cv2.imread(filePath, cv2.IMREAD_GRAYSCALE)
        pictureData = cv2.resize(pictureData, (pictureSize, pictureSize))
        generatedTestData.append([np.array(pictureData), pictureNumber])
    shuffle(generatedTestData)
    np.save('test_Data.npy', generatedTestData)
    return generatedTestData


# If the "training_Data.npy" and "test_Data.npy" files have not been created before:
trainingData = GenerateTrainingData()
testData = GenerateTestData()

# If "training_Data.npy" and "test_Data.npy" files are created:
#trainingData = np.load('training_Data.npy')
#testData = np.load('test_Data.npy')

# we will use it to test 500 formal trainings while training our network.
training = trainingData[:-500]
test = trainingData[-500:]

x_training = np.array([i[0] for i in training]).reshape(-1, pictureSize, pictureSize, 1)
y_training = [i[1] for i in training]
x_test = np.array([i[0] for i in test]).reshape(-1, pictureSize, pictureSize, 1)
y_test = [i[1] for i in test]

### BUILDING THE ARCHITECTURE ###

tf.compat.v1.reset_default_graph()

# let's define what the dimensions of the input of our network will be
convnet = input_data(shape=[None, pictureSize, pictureSize, 1], name='input')

# Convolution layer consisting of 32 pieces of 5x5 filters and relu activation
convnet = conv_2d(convnet, 32, 5, activation='relu')

# max_pool layer of 5x5 filters
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

# Fully connected and relu activated layer consisting of 1024 units
convnet = fully_connected(convnet, 1024, activation='relu')

# dropout layer to prevent overfitting
convnet = dropout(convnet, 0.8)

# Fully connected layer with 2 units and softmax activation
convnet = fully_connected(convnet, 2, activation='softmax')

# build architecture, learning rate, optimization type, loss function and target values ​​we get from filenames
# Let's create the network using #.
convnet = regression(convnet, optimizer='adam', learning_rate=learningRate, loss='categorical_crossentropy',
                     name='targets')


# CREATING A DEEP LEARNING NETWORK (DNN) MODEL WITH THE CREATED ARCHITECTURE
model = tflearn.DNN(convnet, tensorboard_dir='log', tensorboard_verbose=0)

# EDUCATION WITH DATA
model.fit({'input': x_training}, {'targets': y_training}, n_epoch=15,
          validation_set=({'input': x_test}, {'targets': y_test}),
          snapshot_step=500, show_metric=True, run_id=modelName)

### TESTING THE CREATED DEEP NETWORK MODEL ON TEST DATA

fig = plt.figure(figsize=(16, 12))

for no, data in enumerate(testData[:16]):

    pictureNumber = data[1]
    pictureData = data[0]

    y = fig.add_subplot(4, 4, no + 1)
    orig = pictureData
    data = pictureData.reshape(pictureSize, pictureSize, 1)
    networkOutput = model.predict([data])[0]

    if np.argmax(networkOutput) == 1:
        stringLabel = 'Dog'
    else:
        stringLabel = 'Cat'

    y.imshow(orig, cmap='gray')
    plt.title(stringLabel)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()