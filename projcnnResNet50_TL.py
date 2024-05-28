import numpy as np
import tensorflow as tf
import tensorflow.keras as K
from tensorflow.keras.datasets import cifar10
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras import layers
from keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Loading CIFAR-10 data
(X_train, Y_train), (X_test,Y_test) = cifar10.load_data()
X_train,Y_train=shuffle(X_train,Y_train)
X_test,Y_test=shuffle(X_test,Y_test)

# Normalizing
X_train = K.applications.resnet50.preprocess_input(X_train)
X_test = K.applications.resnet50.preprocess_input(X_test)

# One-Hot-Encoding
Y_train_en = to_categorical(Y_train,10)
Y_test_en = to_categorical(Y_test,10)

# Declaring the pretrained model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

# Freeze the convolutional layers
for layer in base_model.layers[:143]:
    layer.trainable = False

# Create a custom classification head
model = Sequential()
model.add(base_model)
# Dropout layers, Flatten layer, Dense layers
model.add(layers.Flatten())
model.add(layers.BatchNormalization())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.BatchNormalization())
model.add(layers.Dense(10, activation='softmax'))

# Combine the base model with the custom head
model.compile(loss ='categorical_crossentropy', optimizer ='adam', metrics =['accuracy'])
history = model.fit(X_train, Y_train_en, epochs = 5,validation_data=(X_test,Y_test_en))
model.summary()

# Extract the accuracy values for training and validation data from the history
acc=history.history['accuracy']
val_acc=history.history['val_accuracy']

# Creating a plot to visualize training and validation accuracy
epochs_acc = range(1, len(acc) + 1)
plt.plot(acc, label='Training acc')
plt.plot(val_acc, label='Validation ACC')
plt.title('Training and Validation ACC')
plt.xlabel('Epochs')
plt.ylabel('ACC')
plt.legend()
plt.show()

# Print test accuracy
loss,req_acc=model.evaluate(X_test,Y_test_en)
print("Test Accuracy:",req_acc)

# Prediction with model
ypred=model.predict(X_test).argmax(axis=1)
print(ypred)
mistake_incides=np.where(ypred!=Y_test)
print(mistake_incides)

# Confusion matrix
mat=confusion_matrix(Y_test,ypred)
plt.figure(figsize=(7, 7))
sns.heatmap(mat, annot=True, fmt='g', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
