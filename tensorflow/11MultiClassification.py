from tfFUNCTIONS import *


"""WORKING WITH A LARGER EXAMPLE (MULTICLASS CLASSIFICATION)"""
#When we have more than two classes then we have multiclass classification
#We will to build a neural network to classify images of different items


#The data has already been sorted into training and test sets
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

#Plot a single sample
plt.imshow(train_data[17])

#Create a small list so we can index onto our training labels so they're human-rediable
classNames = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

#Plot an example image and its label
indexOfChoice = 40000
plt.imshow(train_data[indexOfChoice], cmap=plt.cm.binary)
plt.title(classNames[train_labels[indexOfChoice]])

#Plot multiple random images of fashion dataset
plt.figure(figsize=(7,7))
for i in range(4):
    ax = plt.subplot(2,2, i+1)
    randIndex = random.choice(range(len(train_data)))
    plt.imshow(train_data[randIndex], cmap = plt.cm.binary)
    plt.title(classNames[train_labels[randIndex]])
    plt.axis(False)


#Preprocessing data (Normalization)
train_dataNorm = train_data / train_data.max()
test_dataNorm = test_data / test_data.max()


#Building multi-class classification model
# Input shape = (28*28)
# Output shape = 10
# Softmax

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),             #Makes shape of train_data one-dimensional with Flatten()
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(), #USE THIS WHEN YOUR CODE IS NOT ONE-HOT ENCODED ELSE CATEGORICALCROSSENTROPY()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"]
)

#Callbacks
earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10**(epoch/20))

history = model.fit(train_dataNorm, train_labels, epochs=40, validation_data=(test_dataNorm, test_labels))
#Note: Neural Networks love normalized Data
print(model.evaluate(test_dataNorm, test_labels))
pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Trainingsverlauf")
plt.grid(True)

#Plotting the learning rate decay curve to improve the Learning_Rate
lrs = 1e-3 * (10**(tf.range(40)/20))
plt.semilogx(lrs, history.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")


"""EVALUATING OUR MULTI-CLASS CLASSIFICATION MODEL"""
#Evaluate its perfomance using classifications metrics (confusion matrix)
#Assess/Bewerten some of its predictions (through visualizations)

y_pred = model.predict(test_dataNorm)
y_pred_labels = np.argmax(y_pred, axis=1)

# Klassen-Namen (optional, aber nice)
class_names = [
    "T-Shirt", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"
]

#CONFUSION MATRIX
plot_confusion_matrix_pretty(test_labels, y_pred_labels, class_names=class_names)

#Check out a random image as well as its prediction
plot_random_image(model=model, images=test_data, true_labels=test_labels, classes=class_names)


"""PATTERNS OUR MODEL IS LEARNING / BEHIND THE SCENES OF HIDDEN LAYERS"""
#Find the layers of our most recent model
model.layers                 #low budget version of model.summary()
model.layers[1]              #First Hidden Layer

#Get the patterns of a layer in our network
weights, biases = model.layers[1].get_weights()
biases, biases.shape

plot_model(model, show_shapes=True) #Shows a "Stuktogramm"
























































