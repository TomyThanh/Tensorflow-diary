import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


"""LARGER EXAMPLE WITH A LOT OF DATA"""
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")

#First we have to one-hot encoding our categorical data into numerical data
insuranceEncoded = pd.get_dummies(insurance).astype(float)

# Create X & y values (features and labels)
X = insuranceEncoded.drop("charges", axis = 1)
y = insuranceEncoded["charges"]

# Create training and test sets
from sklearn.model_selection import train_test_split #train_test_split() automatically splits the training and the test sets for us

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
len(X), len(X_train), len(X_test)

### BUILDING A NEURAL NETWORK ###
tf.random.set_seed(42)

insuranceModel = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(1)

])

insuranceModel.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1),
    metrics = ["mae"]
)

#history = insuranceModel.fit(X_train, y_train, epochs = 200)

#Check the results of the insurance model on the test data
evaluation = insuranceModel.evaluate(X_test, y_test) #Bewertet das Modell


#Plot history (also known as loss curve or a training curve
#pd.DataFrame(history.history).plot()
#plt.ylabel("loss")
#plt.xlabel("epoch")
#plt.show()


""" PREPROCESSING DATA (NORMALIZATION and STANDARDIZATION) """
#Neural Networks tend to prefer normalization
#We scale to stabilize the data, some features have higher numbers which the Model thinks is more important
#But all the features are important, so we transform our features into the scale between 0 and 1.

from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

#Create a column transformer
ct = make_column_transformer(
    (MinMaxScaler(), ["age", "bmi","children" ]), #turn all numerical values in these columns between 0 and 1
    (OneHotEncoder(handle_unknown="ignore"), ["sex", "smoker", "region"])
)

#Create X2 & y2 values
X2 = insurance.drop("charges", axis=1)
y2 = insurance["charges"]

#Split train and test set
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

#Fit the column transformer to our training data
ct.fit(X2_train) #Fitting it in our ct = make_column_transformer

#Transform training and test data with normalization (MinMaxScaler) and OneHotEncoder
X2_train_normal = ct.transform(X2_train)
X2_test_normal = ct.transform(X2_test)

#What does our data look like now?
X2_train_normal

#Building Neural Network for our normalized and one-hot encoded data
insuranceModel2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "relu"),
    tf.keras.layers.Dense(1)

])

insuranceModel2.compile(
    loss = tf.keras.losses.MeanAbsoluteError(),
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.1),
    metrics = ["mae"]
)

history2 = insuranceModel2.fit(X2_train_normal, y2_train, epochs = 200)
pd.DataFrame(history2.history).plot()
plt.ylabel("loss")
plt.xlabel("epoch")
plt.show()

