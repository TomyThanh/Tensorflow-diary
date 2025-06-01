import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""SAVING OUR MODELS"""
#Saving our models allows us to use beyond vscode
#1. SavedModel Format
#2. HDF5 Format

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

#Save model using model.save()
model_2.save("best_model.keras") #AS KERAS-FORMAT (RECOMMENDED)


"""LOADING A SAVED MODEL"""
loaded_model = tf.keras.models.load_model("best_model.keras")
loaded_model.summary()
model_2.summary()

#Compare model_2 predictions with KerasModel format model predictions
model_2_preds = model_2.predict(X_test)
loaded_model_preds = loaded_model.predict(X_test)
print(model_2_preds == loaded_model_preds)

