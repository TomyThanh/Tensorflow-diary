import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""EVALUATING MODEL PART 3 / VISUALIZING OF PREDICTIONS PART 2"""

X = tf.range(-100, 100, 4)
y = X + 10

X_train = X[:40]
X_train = tf.reshape(X_train, shape=(-1,1))
y_train = y[:40]
y_train = tf.reshape(y_train, shape=(-1,1))

X_test = X[40:]
X_test = tf.reshape(X_test, shape=(-1,1))
y_test = y[40:]

#1. Create model 
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=[1], name="input_layer1234"),
    tf.keras.layers.Dense(1, name="output_layer")

], name = "my_model")

#2. Compile the model:
model.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ["mae"]

)

model.summary() #SHOWS US A OVERVIEW

#3. Fit the model:
model.fit(X_train, y_train, epochs = 100, verbose=0)
#plot_predictions() SHOWS US THE GRAPH FROM MATPLOTLIB

y_pred = model.predict(X_test)

"""EVALUATING WITH EVALUATION METRICS"""
#MAE - Mean Absolute Error "on average, how wrong is each of my model's predictions"
#MSE - Mean Square Error "square the average errors"


#Evaluate the model on the test
model.evaluate(X_test, y_test)

#Calculate the MAE
mae = tf.metrics.MeanAbsoluteError()   # 1. Initialisieren
mae.update_state(y_test, y_pred)       # 2. Werte einfüttern
result = mae.result().numpy()          # 3. Ergebnis abrufen
print(f"MAE: {result}")

#Calculate the MSE
mse = tf.metrics.MeanSquaredError()    # 1. Initialisieren
mse.update_state(y_test, y_pred)       # 2. Werte einfüttern
result2 = mse.result().numpy()         # 3. Ergebnis abrufen
print(f"MSE: {result2}")


# Make some functions to reuse MAE and MSE
def mae(y_true, y_pred):
    mae = tf.metrics.MeanAbsoluteError()
    mae.update_state(y_true, tf.squeeze(y_pred))
    result = mae.result().numpy()
    return result

def mse(y_true, y_pred):
    mse = tf.metrics.MeanSquaredError()
    mse.update_state(y_true, tf.squeeze(y_pred))
    result = mse.result().numpy()
    return result

