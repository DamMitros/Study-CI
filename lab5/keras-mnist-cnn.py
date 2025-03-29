import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import History

# Load dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess data

# a) W preprocessingu, dane są przygotowywane do użycia w modelu sieci neuronowej
# Funkcja reshape jest używana do zmiany kształtu danych, aby pasowały do oczekiwanego formatu wejściowego modelu.
# Funkcja to_categorical jest używana do kodowania etykiet klas w postaci "one-hot".
# Funkcja np.argmax jest używana do odwrócenia kodowania "one-hot" i uzyskania pierwotnych etykiet.

train_images = train_images.reshape((train_images.shape[0], 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
original_test_labels = np.argmax(test_labels, axis=1)  # Save original labels for confusion matrix

# Define model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)), # b) wchodzi znormalizowane zdjęcie 28x28x1
    MaxPooling2D((2, 2)), # wchodzi zdjęcie 28x28x1, wychodzi 14x14x32
    Flatten(), # wchodzi zdjęcie 14x14x32, wychodzi 6272 elementów
    Dense(64, activation='relu'), # wchodzi 6272, wychodzi 64 neurony
    Dense(10, activation='softmax') # wchodzi 64, wychodzi 10 neuronów (klas)
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = History()
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2, callbacks=[history])

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")

# Predict on test images
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)

# Confusion matrix

# c) Najwięcej błędów w macierzy błędów dotyczy par cyfr, które są do siebie podobne.
# Pary (2, 7) (4, 9) (3, 5) są często mylone. 
cm = confusion_matrix(original_test_labels, predicted_labels)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')
# plt.show()

# Plotting training and validation accuracy

# d) Krzywe uczenia się mogą dostarczyć informacji na temat przeuczenia lub niedouczenia się modelu.
# Jeśli krzywa dokładności walidacji zaczyna się oddzielać od krzywej dokładności treningu, może 
# to wskazywać na przeuczenie. W tym przypadku możemy dojść do wnioski nieznacznego przeuczenia modelu.
# (train accuracy jest wyższe niż validation accuracy)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', color='grey')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curve.png')
# plt.show()

# e) Aby zapisać model sieci do pliku h5 co epokę, pod warunkiem, że w tej epoce osiągnęliśmy lepszy wynik,
# można użyć funkcji zwrotnej ModelCheckpoint z biblioteki keras.callbacks.
# Funkcja ta może być wywoływana podczas treningu modelu i zapisywać model do pliku w każdej epoce, jeśli osiągnięto lepsze wyniki.
# Potrzebna jest interakcja z modelem podczas treningu.

# from tensorflow.keras.callbacks import ModelCheckpoint