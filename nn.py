import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras as ks
import sklearn as sk
import matplotlib.pyplot as plt
import random as rd

ds = tf.keras.datasets.mnist
# 28x28, hand-written digits. (MNIST/EMNIST/MNIST_CORRUPTED/CMATERDB for TFDS.)

(xtr, ytr), (xte, yte) = ds.load_data()
# Load data from the provided dataset.

xtr = tf.keras.utils.normalize(xtr, axis = 1) 
xte = tf.keras.utils.normalize(xte, axis = 1)
# Normalize said data.

mdl = tf.keras.models.Sequential()
mdl.add(tf.keras.layers.Flatten())
mdl.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
mdl.add(tf.keras.layers.Dense(128, activation = tf.nn.relu))
mdl.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))
# Set params for 'mdl'.

mdl.compile(optimizer = 'adam',
            loss = 'sparse_categorical_crossentropy',
            metrics = ['accuracy'])
# Compile 'mdl' with provided optimizer. | In this case the optimizer is 'adam'.

# mdl.fit(xte, yte, epochs = 50) # Can be removed if not needed.
# Train the model on the provideded dataset. | In this case it will train for 50 epochs.

# vl, va = mdl.evaluate(xte, yte) # Can be removed if not needed.
# Evaulate accuracy of the neural network.

# mdl.save("#") (Save the dataset.) # Can be removed if not needed.
newmdl = tf.keras.models.load_model('#')
# Save and load the trained dataset for the NN.

prediction = newmdl.predict([xte])
# Makes the prediction function so it can be called upon.

rndm = rd.randint(0, 10000)

def pd():
    print(np.argmax(prediction[rndm]))
    plt.imshow(xte[rndm])
    plt.show()
    # Use a random integer between 0 - 10,0000 and then predict what digit the image is based on the dataset entry of the random int.

while True:
    inp = input("Generate? (y/n)?\n").lower()
    if inp in ['y', 'yes']:
        rndm = rd.randint(0, 10000)
        pd()
    elif inp in ['n', 'no']:
        print("Thank you for using a program made by Malik Hassan!")
        exit()
    else:
        print("Invalid input. Please enter 'y', 'yes', 'n', or 'no'.")
# Make sure that you want to generate to free up memory.