import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

"""CREATING TENSORS"""

#Tensors sind wie Arrays
#tf.constant
nixTensor = tf.constant([10,10]) #Erstellt ein unveränderbares Tensor
matrixTensor = tf.constant([[10, 7], [7, 10]])
floatMatrixTensor = tf.constant([[10.1,7.],
                                 [3., 2.],
                                 [8.2, 6.7]], dtype= tf.float32)
rangeTensor = tf.range(1,11) 

#tf.Variable
veraenderbarTensor = tf.Variable([10,7]) #Erstellt ein veränderbares Tensor
stringTensor = tf.Variable("Das ist ein String", tf.string) #Erstellt ein veränderbares Tensor
nummerTensor = tf.Variable(123, tf.int16)  #Erstellt ein veränderbares Tensor 
rank2Tensor = tf.Variable([["Test", "ok"], ["test", "ok"], ["TEST", "OK"], ["ok", "ok"]], tf.string)
onesTensor = tf.ones([5,5])
zerosTensor = tf.zeros([5,5,5,5])


#tf.Variable Tensor Werte verändern
veraenderbarTensor[0].assign(7) #verändert den ersten Wert zu 7


#random Tensor Normalverteilung
randomTensor1 = tf.random.Generator.from_seed(42) 
randomTensor1 = randomTensor1.normal(shape=(3,2))
randomTensor2 = tf.random.Generator.from_seed(42)
randomTensor2 = randomTensor2.normal(shape=(3,2))


#Shuffle Tensor (mischt die Werte in ein Tensor komplett durch)
notShuffled = tf.constant([[10,7],
                           [3,4],
                           [2,5]])
shuffledTensor = tf.random.shuffle(notShuffled)

#Wenn globale und operation seed gesetzt wurden, dann werden beide seeds verwendet um die Zufallssequenz zu bestimmen
tf.random.set_seed(42)
notShuffled2 = tf.constant([[9,8],
                           [2,3],
                           [1,6]])
shuffledTensor2 = tf.random.shuffle(notShuffled2, seed=42)


#Scalar: a single number
#Vector: one-dimensional array
#Matrix: 2-dimensional array
#Tensor: n-dimensional array

"""METHODS TO TENSORS"""

#Ranks und Dimensionen
array = np.array([[2,3],[3,4]])
arrayTensor = tf.Variable(array) #Ranks sind die Dimensionen, die ein Tensor hat
rank = tf.rank(arrayTensor) #Bestimmt den Rank des Tensors
matrixTensorDimension = matrixTensor.ndim #Gibt die Dimension von matrixTensor


#Numpy Array in Tensor
array2 = np.arange(1,25)
numpyTensor = tf.constant(array2, shape=(2,3,4))


#Reshape
shape = rank2Tensor.shape #Gibt den shape an
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1, [2,3,1]) #reshape tensor1 in 2,3,1
tensor3 = tf.reshape(tensor2, [3, -1]) #reshape tensor2 in 3 und beste
zerosReshape = tf.reshape(zerosTensor, [125,-1])

#Wichtige Informationen/Attribute zu Tensors
print(f"Datatype of every element: {zerosTensor.dtype}")
print(f"Number of dimensions (rank): {zerosTensor.ndim}")
print(f"Shape of tensor: {zerosTensor.shape}")
print(f"Elements along the 0 axis: {zerosTensor.shape[0]}")
print(f"Elements along the last axis: {zerosTensor.shape[-1]}")
print(f"Total number of elements in our tensor: {tf.size(zerosTensor).numpy()}")


#Arten von Tensors
#1 Variables veränderbar
#2 Constant nicht veränderbar
#3 Placeholder nicht veränderbar
#4 SpareTensor nicht veränderbar

#Überprüfung von Tensoren mit der Funktion
#with tf.Session() as sess:
    #tensor1.eval()
