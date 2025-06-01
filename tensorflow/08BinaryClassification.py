import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_circles


"""INTRODUCTION to NEURAL NETWORK CLASSIFICATION with TENSORFLOW"""
# 1. Binary classification
# 2. Multiclass classification
# 3. Multilabel classification


nSamples = 1000

#Create circle / DATASET
X, y = make_circles(nSamples, noise= 0.03, random_state=42)

#Visualizing data
circle = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
plt.scatter(X[:, 0], X[:, 1], c=y, cmap = plt.cm.RdYlBu)

X.shape, y.shape

#Creating a neural Network
modelAZ = tf.keras.Sequential([

    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dropout(0.2),

    tf.keras.layers.Dense(32, activation = "relu"),
    tf.keras.layers.Dense(1, activation = "sigmoid")

])

modelAZ.compile(
    loss = tf.keras.losses.BinaryCrossentropy(), #WE HAVE TO USE THIS FOR BINARY CLASSIFICATION
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
    metrics = ["accuracy"] #HOW ACCURATE YOUR MODEL IS

)

#Callback
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20,
    restore_best_weights=True
)

history = modelAZ.fit(X, y, epochs = 1000, validation_split = 0.2 ,callbacks = [early_stop])
print(modelAZ.evaluate(X, y))

pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Trainingsverlauf")
plt.grid(True)
plt.show()

#Visualizing model's predictions with a COMPLEX FUNCTION
def plot_decision_boundary(model, X, y):
    """Plots the decision boundary created by a model predicting on X"""
    X_min, X_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(X_in)
    y_pred = np.reshape(y_pred, xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), xx.max())
    plt.show()

plot_decision_boundary(modelAZ, X=X, y=y)


