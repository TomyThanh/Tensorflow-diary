import os
import itertools
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.datasets import make_circles
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.datasets import fashion_mnist
from keras.utils import plot_model
import random

##Visualizing model's predictions with a COMPLEX FUNCTION
def plot_decision_boundary(model, X, y):
    """Plots the decision boundary created by a model predicting on X"""
    X_min, X_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(X_min, X_max, 100),
                         np.linspace(y_min, y_max, 100))

    X_in = np.c_[xx.ravel(), yy.ravel()]
    y_pred = model.predict(X_in)
    y_pred = np.reshape(y_pred, xx.shape)
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:,1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), xx.max())
    plt.show()


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


#replicating sigmoid-function
def sigmoid(x):
    return 1/(1+tf.exp(-x))

#replicating relu-function
def relu(x):
    return tf.maximum(0, x)

#CONFUSION MATRIX
def plot_confusion_matrix_pretty(y_true, y_pred, class_names=None, normalize=True, figsize=(10,10), cmap=plt.cm.Blues):
    """
    Zeigt eine formatierte Confusion Matrix mit Optionen für Normalisierung und Klassenbeschriftung.

    Parameters:
    - y_true: array-like, true labels
    - y_pred: array-like, predicted labels (kann probabilistisch sein – wird gerundet)
    - class_names: list of class names (default: nummerisch)
    - normalize: bool, ob Matrix normalisiert werden soll
    - figsize: tuple, Größe der Darstellung
    - cmap: Farbkarte
    """

    # Vorhersagen ggf. runden (bei probabilistischen Outputs)
    # Vorhersagen: Wahrscheinlichkeiten zu Klassenlabels machen
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred = np.argmax(y_pred, axis=1)

    # Confusion Matrix berechnen
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] if normalize else cm
    n_classes = cm.shape[0]

    # Klassenlabels
    labels = class_names if class_names else np.arange(n_classes)

    # Plot Setup
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.matshow(cm, cmap=cmap)
    fig.colorbar(cax)

    ax.set(title="Confusion Matrix",
           xlabel="Predicted Label",
           ylabel="True Label",
           xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=labels,
           yticklabels=labels)

    # Schwellenwert für Textfarbe
    threshold = (cm.max() + cm.min()) / 2.

    # Zahlen & Prozentwerte eintragen
    for i, j in itertools.product(range(n_classes), range(n_classes)):
        value = f"{cm[i, j]}"
        if normalize:
            value += f"\n({cm_norm[i, j]*100:.1f}%)"
        ax.text(j, i, value,
                horizontalalignment="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=13)

    plt.tight_layout()
    plt.show()


#Picks a random image, plots it and labels it with a prediction and truth label
def plot_random_image(model, images, true_labels, classes):
    # Zufälliges Bild auswählen
    i = random.randint(0, len(images) - 1)  # Korrekte Range

    target_image = images[i]  # Das zufällige Bild
    target_label = true_labels[i]  # Das wahre Label

    # Vorhersage des Modells
    target_image = target_image.astype("float32")  # Sicherstellen, dass das Bild den richtigen Datentyp hat
    target_image = np.expand_dims(target_image, axis=0)  # Bild auf Batch-Größe 1 erweitern
    pred_probs = model.predict(target_image)  # Modellvorhersage
    pred_label = classes[pred_probs.argmax()]  # Vorhersage basierend auf den Wahrscheinlichkeiten

    # Zurückskalieren der Bildwerte von [0, 1] auf [0, 255]
    target_image = (target_image[0] * 255).astype(np.uint8)

    # Visualisierung des Bildes
    plt.imshow(target_image)
    
    # Setze die Farbe basierend darauf, ob die Vorhersage korrekt ist
    if pred_label == classes[target_label]:
        color = "green"
    else:
        color = "red"

    # Zeige Vorhersage und wahres Label im Titel an
    plt.xlabel("Pred: {} {:2.0f}% (True: {})".format(pred_label, 100*tf.reduce_max(pred_probs), classes[target_label]),
               color=color)  # Wenn die Vorhersage richtig oder falsch ist, wird die Farbe angepasst
    plt.show()


#Konvertiert ein TensorFlow Dataset in NumPy-Arrays
def dataset_to_numpy(dataset):
    images = []
    labels = []
    
    for image, label in dataset:
        images.append(image.numpy())
        labels.append(label.numpy())
    
    images = np.concatenate(images, axis=0)
    labels = np.concatenate(labels, axis=0)
    
    return images, labels

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def predict_custom_image(image_path, model, class_names, img_size=(224, 224)):
    # Bild laden
    img = Image.open(image_path).convert("RGB")
    
    # Größe anpassen
    img_resized = img.resize(img_size)
    
    # In Array umwandeln & normalisieren
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen

    # Vorhersage
    pred_probs = model.predict(img_array)
    pred_index = np.argmax(pred_probs)
    pred_label = class_names[pred_index]
    confidence = 100 * np.max(pred_probs)

    # Ergebnis anzeigen
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"Vorhersage: {pred_label} ({confidence:.2f}%)", color="green" if confidence > 60 else "orange")
    plt.show()

    return pred_label, confidence

