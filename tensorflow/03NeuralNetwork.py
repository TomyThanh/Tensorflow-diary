import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


"""EVALUATING MODEL PART 2 / VISUALIZING THE MODEL ITSELF"""

#EXAMPLE: Building neural network for training data and test data

X = tf.range(-100, 100, 4)
y = X + 10

X_train = X[:40]
X_train = tf.reshape(X_train, shape=(-1,1))
y_train = y[:40]
y_train = tf.reshape(y_train, shape=(-1,1))

X_test = X[40:]
X_test = tf.reshape(X_test, shape=(-1,1))
y_test = y[40:]

#1: Create model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])

#2: Compile model
model.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01),
    metrics = ["mae"]
)

#3: Fit model
#model.fit(X_train, y_train, epochs=100)


"""VISUALIZING OUR MODEL"""
#model.summary() shows us useful information for evaluation


#Creating a model that builds automatically by defining the input_shape argument in the first layer
tf.random.set_seed(42)

#1. Create model (same as above)
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

model.summary()

#1. Total params - total number of parameters in model
#2. Trainable parameters - these are the parameters (patterns) the model can update as it trains
#3. Non-trainable params - these parameters aren't updated during training
#Non-trainable params are typical when we bring in already learned patterns from other models (transfer learning)

#3. Fit the model:
model.fit(X_train, y_train, epochs = 100, verbose=0)

#VISUALIZE MODEL WITH TF.PLOT-MODEL
from tensorflow.keras.utils import plot_model
plot_model(model, to_file="model_plot.png", show_shapes=True) #Creates a overview in a png of our model


"""VISUALIZING THE PREDICTION PART 1"""
# "y_test" vs. "y_pred" (ground truth vs your model)

#Make some predictions
y_pred = model.predict(X_test)

#Plotting-function to visualize our predictions!!!
def plot_predictions(train_data = X_train, train_labels = y_train,
                     test_data = X_test, test_labels = y_test, predictions = y_pred):

#Plots training data, test data and compares predictions to ground truth label
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", label = "Training data")
    plt.scatter(test_data, test_labels, c="g", label= "Testing data")
    plt.scatter(test_data, predictions, c="r", label="Predictions")
    plt.legend()
    plt.show()

plot_predictions(train_data = X_train,
                 train_labels= y_train,
                 test_data= X_test,
                 test_labels= y_test,
                 predictions= y_pred)
