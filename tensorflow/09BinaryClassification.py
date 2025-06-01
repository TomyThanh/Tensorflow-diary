import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_circles
from tfFUNCTIONS import *
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

"""NON-LINEARITY FOR CLASSIFICATION (BINARY)"""
#Note: The combination of linear and non-linear functions is very important 

nSamples = 1000
X, y = make_circles(nSamples, noise= 0.03, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation = "relu"),
    tf.keras.layers.Dense(4, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid") #VERY IMPORTANT USE "SIGMOID" FOR BINARY CLASSIFICATION
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate= 0.001),
    metrics = ["accuracy"]
)

#history = model.fit(X, y, epochs = 400)
model.evaluate(X,y)

#Check the decision boundary
#plot_decision_boundary(model = model, X=X, y=y)


"""Non-linear functions and linear functions / EXAMPLE"""
A = tf.cast(tf.range(-10,10), tf.float32)

#replicating sigmoid-function
def sigmoid(x):
    return 1/(1+tf.exp(-x))

plt.plot(sigmoid(A))

#replicating relu-function
def relu(x):
    return tf.maximum(0, x)
plt.plot(relu(A))


#IT IS IMPORTANT TO KNOW THAT LINEAR (tf.keras.activations.linear) LITERALLY DOES NOTHING
#THAT IS WHY WE USE NON-LINEAR FUNCTIONS (SIGMOID & RELU)


"""IMPROVEMENT"""
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model2 = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model2.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.004),
    metrics=["accuracy"]
)

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))

history2 = model2.fit(X_train, y_train, epochs=1000, validation_split = 0.2, callbacks=[early_stop])
print(model2.evaluate(X_test, y_test))

pd.DataFrame(history2.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Trainingsverlauf")
plt.grid(True)
plt.show()

plot_decision_boundary(model=model2, X=X_test, y=y_test)


"""LEARNING RATE IMPROVEMENT"""
#Finding the learning rate where the loss decreases the most
# 1. learning-rate callback

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
#Plot the learning curve and then look at the most decreasing point and check for the y-value of learning_rate
#You will have to experiment with lr_scheduler. Afer that REMOVE IT FROM YOUR CALLBACKS.

#Plotting the learning rate decay curve
lrs = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrs, history2.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.show()



































