import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


"""AGGREGRATING TENSORS"""
#Aggregrating tensors = condensing them from multiple values down to a smaller amount of values

dc = tf.constant([-7, -10])
tf.abs(dc) #Absolute values = Betrag von einem Wert

#Forms of aggregration:
#1: Get the minimum
#2: Get the maximum
#3: Get the mean(durchschnitt)
#4: Get the sum

#Example
ex = tf.constant(np.random.randint(1,101, size=50))


tf.reduce_min(ex) #get the minimum
tf.reduce_max(ex) #get the maximum
tf.reduce_mean(ex) #get the mean/average
tf.reduce_sum(ex) #get the the sum

ex = tf.cast(ex, dtype=tf.float32)
tf.math.reduce_variance(ex) #get the variance but  dtype must be float
tf.math.reduce_std(ex) #get the standard deviation but dtype must be float


#FIND THE POSITIONAL MAXIMUM AND MINIMUM
tf.random.set_seed(42)
fy = tf.random.uniform(shape=[50])

#find the positional maximum or minimum
tf.argmax(fy)
tf.argmin(fy)

#index on our largest/smallest value position
fy[tf.argmax(fy)] == tf.reduce_max(fy) #both are equal
fy[tf.argmin(fy)] == tf.reduce_min(fy) #both are equal


#SQUEEZING A TENSOR (REMOVING ALL SINGLE DIMENSIONS)"""
tf.random.set_seed(42)
gf = tf.constant(tf.random.uniform(shape=[50]), shape = (1,1,1,1,50))
gfSqueezed = tf.squeeze(gf) #Entfernt alle 1-dimensionale Dimensions


"""ONE-HOT ENCODING TENSORS"""
someList = [0,1,2,3]

#One-hot encoding wandelt Liste in Binär-Tensor (mit Vektoren), sodass Computer es versteht
tf.one_hot(someList, depth=4) #Depth ist die Anzahl an 0 (0=aus, 1 = an)


"""TENSORFLOW & NUMPY"""
#Numpy und Tensorflow sind sehr gut kombinierbar. Man kann Tensors in Arrays umwandeln aber auch umgekehrt

jj = tf.constant(np.array([3.,7.,10.]))
np.array(jj) #From Tensor to Numpy-Array
jj.numpy() #Alternative

#Numpy Arrays benutzen andere Dtypes (float64) als Tensors(float32)
