import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np


"""INDEXING"""

zerosTensor = tf.zeros((2,3,4,5))

# Get the first 2 elements of each dimension
zerosTensor[:2,:2,:2,:2]


# Get the first from each dimension except the final one
zerosTensor[:1,:1,:1]


#Get the last item of each of row
rank2Tensor = tf.constant([[10, 7], 
                           [3, 4]])
rank2Tensor[:,-1]


#Add in extra dimension to rank2 tensor
rank3Tensor = rank2Tensor[..., tf.newaxis] #tf.newaxis adds another dimension


#Alternative to tf.newaxis
tf.expand_dims(rank2Tensor, axis = -1) # "-1" means expand the final axis "0" = first axis, "1" = middle axis


"""MATRIX MULTIPLICATION / DOT PRODUCT"""

#Basics operations = Arithmetische Operatoren
tensor = tf.constant([[10, 7], [3,4]])
tensorAddition = tensor + 10
tensorMultiplikation = tensor * 10
tensorSubtraktion = tensor - 10
tensorDivision = tensor / 10

#Matrizenmultiplikation / Matrix multiplication
tf.matmul(tensor, tensor)
tf.tensordot(tensor,tensor, axes=1)

#Matrizenmultiplikation mit Python operator "@"
tensor @ tensor

#Regel zu Matrizenmultiplikation
#1: Die innere Dimensionen müssen übereinstimmen
#2: Ergebnismatrix hat den gleichen Shape wie die äußeren Dimensionen

x = tf.constant([[1,2], [3,4], [5,6]]) #Shape (3,2)
y = tf.constant([[7,8], [9,10], [11, 12]]) #Shape (3,2)

y = tf.reshape(y, shape=(2,3)) #Hier müssen wir den shape ändern für die Übereinstimmung 
x @ y #(3,2) * (2,3) innere Dimensionen übereinstimmen und Ergebnis bekommt äußere dimension also 3,3

#tf.transpose
tf.transpose(x)  #Wechselt die Zeilen mit Spalten, kann auch shape verändern

#Dot product (Skalarprodukt) is also reffered to Matrix Multiplication
y = tf.constant([[7,8], [9,10], [11, 12]]) 
tf.tensordot(tf.transpose(x),y,axes=1) #Alternative zu tf.matmul

tf.matmul(x, tf.reshape(y, shape=(2,3)))


"""CHANGING DATATYPE"""
b = tf.constant([1.7, 7.4])
c = tf.constant([7,10])

#Change from float32 to float16 (reduced precision but higher speed)
d = tf.cast(b, dtype=tf.float16) #ändert dtype in float16


#Change from int32 to float32
e = tf.cast(c, dtype=tf.float32)
