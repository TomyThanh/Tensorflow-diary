
"""IMPROVING FIRST MODEL"""
#Improvements can be made in creating a model, compiling a model and fitting a model / CHANGING HYPERPARAMETER

#1: Creating a model:  -add more layers
#                      -increase number of hidden units (neurons)
#                      -change activation function in each layer

#2: Compiling a model:  -change optimization function (Adam > SGD)
#                       -change learning rate (MOST IMPORTANT THING!!!!)

#3: Fitting a model:  -more epochs
#                     -more data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


#Improvement by adding more Hidden Layers with more Neurons and using Adam

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0 , 14.0])
y = X + 10
X = tf.constant(X, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)
X = tf.reshape(X, shape=(-1,1)) #VERY IMPORTANT RESHAPE IT TO 2-DIMENSIONAL

#1: Create the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = "relu"), #Increasing Neurons and using activation function "relu"
    tf.keras.layers.Dense(1)

])

#2: Compile the model
model.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01), #Changing Optimization Function to Adam and learning rate
    metrics = ["mae"]

)

#3: Fit the model
#model.fit(X, y, epochs=100)
prediction = tf.constant([17.0])




"""EVALUATING THE MODEL PART 1"""
#Build a model -> fit it -> evaluate it -> tweak a model
#IMPORTANT: VISUALIZE, VISUALIZE, VISUALIZE!!!!

#The data - what data are working with?
#The training model - how does model perform?
#The predictions of model- how far is the prediction to the truth?


#SECOND EXAMPLE
X = tf.range(-100, 100, 4)
y = X + 10

#Visualize the data
plt.scatter(X,y)


#THE 3 SETS:
# Training set - the model learns from this data 70-80% of total data
# Validation set - the model gets tuned on this data, 10-15% of total data
# Test set - the model gets evaluated on this data to test, 10-15% of total data


#Check the length of how many samples we have
len(X)

#Split the data into train and test set
X_train = X[:40] #Frist 40 are training sample (80%)
y_train = y[:40]

X_test = X[40:] #Last 10 are test sample (20%)
y_test = y[40:]

#Visualizing the data in training and test sets
#Plot training data in blue
plt.figure(figsize=(10,7))
plt.scatter(X_train, y_train, c="b", label="Training data")
#Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Testing data")
#Show a legend
plt.legend()
plt.show()
