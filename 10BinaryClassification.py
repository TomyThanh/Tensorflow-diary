from tfFUNCTIONS import *

"""TRAINING AND EVALUATING A MODEL WITH AN IDEAL LEARNING_RATE"""
nSamples = 1000
X, y = make_circles(nSamples, noise= 0.03, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(64, activation = "relu"),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.02), #IDEAL LEARNING_RATE WE'VE FOUND THIS WITH LR_SCHEDULER
    metrics = ["accuracy"]
)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
history = model.fit(X_train, y_train, epochs = 30, validation_split = 0.2, callbacks = [early_stop])
print(model.evaluate(X_test,y_test))

pd.DataFrame(history.history).plot()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.title("Trainingsverlauf")
plt.grid(True)
plt.show()

plot_decision_boundary(model=model, X=X_test, y=y_test)

"""MORE CLASSIFICATION EVALUATION METHODS"""
# Accuracy - default and basic
# Precision - avoiding false positives
# Recall - avoiding false negatives
# F1-Score - Combination of Precision and Recall

#How about a CONFUSION MATRIX?
y_pred = model.predict(X_test)
print(confusion_matrix(y_true=y_test, y_pred=tf.round(y_pred)))


#How about we prettify our confusion matrix?
figsize = (10,10)
cm = confusion_matrix(y_test, tf.round(y_pred))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] #Normalize our confusion matrix

plot_confusion_matrix_pretty(y_test, y_pred, class_names=["Negativ", "Positiv"]) #THIS FUNCTION PLOTS THE CONFUSION MATRIX

