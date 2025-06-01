import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from _03NeuralNetwork import *
from _04NeuralNetwork import *


"""EXPERIMENTING"""
### RUNNING EXPERIMENTS TO IMPROVE OUR MODEL
# Build a model -> fit it -> tweak it -> repeat

#Let's do 3 modelling experiments:
#1. "model 1" - same as the original model, 1 layer, trained for 100 epochs
#2. "model 2" - 2 layers, trained for 100 epochs
#3. "model 3" - 2 layers, trained for 500 epochs


## Build "model 1"

#Data
X = tf.range(-100, 100, 4)
y = X + 10

X_train = X[:40]
X_train = tf.reshape(X_train, shape=(-1,1))
y_train = y[:40]
y_train = tf.reshape(y_train, shape=(-1,1))

X_test = X[40:]
X_test = tf.reshape(X_test, shape=(-1,1))
y_test = y[40:]

#1. Create the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
    ])

#2. Compile the model
model_1.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01),
    metrics = ["mae"]
)

#3. Fit the model
model_1.fit(X_train, y_train, epochs = 100)

#Make and plot predictions for model_1
y_preds_1 = model_1.predict(X_test)
plot_predictions(predictions= y_preds_1)

#Calculate model_1 evaluation metrics
mae_1 = mae(y_test, y_preds_1)
mse_1 = mse(y_test, y_preds_1)


#Build "model_2"
# 2 Dense layers, trained for 100 epochs
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ["mae"]
)

model_2.fit(X_train, y_train, epochs = 100)
y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)

#Calculate model_2 evaluation metrics
mae_2 = mae(y_test, y_preds_2)
mse_2 = mse(y_test, y_preds_2)


#Build "model_3"
#2 layers, trained for 500 epochs

model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)

])

model_3.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ["mae"]
)

model_3.fit(X_train, y_train, epochs = 500)
y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions= y_preds_3)

#Calculate model_3 evaluation metrics
mae_3 = mae(y_test, y_preds_3)
mse_3 = mse(y_test, y_preds_3)

#Note: You want to start with small experiments and make sure they work

"""COMPARING THE RESULTS OF OUR EXPERIMENTS """

#Let's compare our model's results using pandas DataFrames
import pandas as pd

model_results = [["model_1", mae_1, mse_1],
                 ["model_2", mae_2, mse_2],
                 ["model_3", mae_3, mse_3]]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results)

###Tracking your experiments
#Tracking the results of your experiments is very good
#Tools for helping us:
#1. TensorBoard - a component of the tensorflow library helping track modelling experiments
#2. Weights & Biases - a tool for tracking all kinds of ML experiments (compatible with TensorBoard)
