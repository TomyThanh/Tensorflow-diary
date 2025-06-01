"""INTRODUCTION TO REGRESSION NEURAL NETWORK"""

#Predicting a number based on other numbers

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

#Creating features
X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0 , 14.0])

#Create labels
y = np.array([3.0, 6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0])

#Visualize it
plt.scatter(X, y)
#plt.show() zeigt das Diagramm an

#input and output shapes
#Create a demo tensor for our housing problem
houseInfo = tf.constant(["bedroom", "bathroom", "garage"])
housePrice = tf.constant([939700])


#Turn our NumPy array into tensors
X = tf.constant(X)
y = tf.constant(y)
X = tf.reshape(X, (-1, 1)) #VERY IMPORTANT TENSOR HAS TO BE 2-DIMENSIONAL SO THE MODEL CAN TRAIN!!!!!!!!!!!!




#Steps in modelling with TensorFlow
#1: ℂ𝕣𝕖𝕒𝕥𝕚𝕟𝕘 𝕒 𝕞𝕠𝕕𝕖𝕝 - define the input and output layers and hidden layers
#2: ᴄᴏᴍᴘɪʟᴇ ᴀ ᴍᴏᴅᴇʟ - define loss function (tells how wrong model is) and optimizer (tells model how to improve)
#3: FIƬƬIПG Λ MӨDΣL - letting the model try to find patterns between X & y


"""EXAMPLE / FIRST MODEL"""
tf.random.set_seed(42)

X = np.array([-7.0, -4.0, -1.0, 2.0, 5.0, 8.0, 11.0 , 14.0])
y = X + 10
X = tf.constant(X, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)
X = tf.reshape(X, shape=(-1,1)) #VERY IMPORTANT RESHAPE IT TO 2-DIMENSIONAL


#1. Create a model using the Sequential API
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1)
])
#Alternative model.add(tf.keras.layers.Dense(1))


#2. Compile the model
model.compile(
            loss = tf.keras.losses.MeanAbsoluteError(), #mae = mean absolute error
            optimizer = tf.keras.optimizers.SGD(learning_rate = 0.01), #SGD = stochastic gradient descent
            metrics=["mae"])  


#3. Fit the model
model.fit(X, y, epochs= 5)


#4. Make a Prediction using our model
prediction = np.array([17.0]) #IMPORTANT MUST BE NP-ARRAY BECAUSE X IS ALSO NP ARRAY / TENSOR
print(model.predict(prediction))
