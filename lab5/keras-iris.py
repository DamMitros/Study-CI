import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

# Load the iris dataset
iris = load_iris()
X = iris.data
y = iris.target
# Preprocess the data
# Scale the features

# a) StandardScaler transformuje dane tak, aby miały one średnią 0 i wariancję 1
# Jest to robione, aby znormalizować dane i uniknąć sytuacji, w której wybrane zmienne 
# zdominują nad innymi zmiennymi. 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Encode the labels

#b)OneHotEncoder transformuje etykiety klas na wektory binarne. Każda klasa jest reprezentowana
#przez unikatowy wektor, w którym dokładnie jedna wartość wynosi 1, a reszta to 0. Kodowanie one-hot
#eliminuje błędną interpretację porządku między kategoriami, która mogłaby się pojawić przy użyciu 
#wartości liczbowych. Przykład:
# klasy = banan, pomaarńcza, jabłko
# po transformacji: [1, 0, 0], [0, 1, 0], [0, 0, 1]

encoder = OneHotEncoder(sparse_output=False)
# encoder = OneHotEncoder(sparse=False)
y_encoded = encoder.fit_transform(y.reshape(-1, 1))
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.3,
random_state=42)
# Define the model
model = Sequential([

# c) Warstwa wejściowa ma 4 neurony, ponieważ zmienna X_train ma 4 kolumny. X_train.shape[1] oznacza
# liczbę kolumn w X_train. Warstwa wyjściowa ma 3 neurony, ponieważ zmienna y_encoded ma 3 kolumny.
# Ogólem model powinien mieć tyle neuronów w warstwie wejściowej ile kolumn jest danych, z którymi
# model ma pracować. Warstwa wyjściowa powinna mieć tyle neuronów ile jest klas w danych.

Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
Dense(64, activation='relu'),
Dense(y_encoded.shape[1], activation='softmax')

# d) Funkcja aktywacji sigmoid => dokładność 88.89%
# Funkcja aktywacji tanh => dokładność 97.78%
# Funkcja aktywacji relu => dokładność 100%
# Funkcja aktywacji softmax => dokładność 71.11%
# Aktywacja "softmax" normalizuje wyniki do przedziału (0,1) - rozkład prawdopodobieństwa
# Najlepszy wynik uzyskano dla aktywacji relu
])
# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from tensorflow.keras.optimizers import Adam, SGD, RMSprop

# Dla Adam z niestandardową szybkością uczenia -- 100%
# optimizer = Adam(learning_rate=0.1)
# Adam połączenie SGD i RMSprop. Jest to bardziej zaawansowany optymalizator, który
# automatycznie dostosowuje szybkość uczenia się na podstawie pierwszego i drugiego momentu gradientu. (uniwersalny)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Dla SGD z niestandardową szybkością uczenia i momentem -- 86,67%
# SGD optymalizuje funkcję straty, aktualizując wagi modelu na podstawie gradientu funkcji straty (dokładny, powolny)
# optimizer = SGD(learning_rate=0.01)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# Dla RMSprop z niestandardową szybkością uczenia -- 35.56% / 73.33% / 64,44% -- zbyt mała szybkość uczenia
# RMSprop jest podobny do SGD, ale dostosowuje szybkość uczenia się dla każdej wagi na podstawie
# średniej kwadratowej gradientu. Jest to bardziej efektywne w przypadku danych o dużej wariancji. (bardziej elastyczny)
# optimizer = RMSprop(learning_rate=0.0001)
# model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# e) Różne optymalizatory i funkcje straty dają różne wyniki. Możemy dostosować szybkość uczenia się w optymalizatorze.
# Tak jak widać najlepszy wynik uzyskano dla optymalizatora Adam. Dla SGD i RMSprop uzyskano gorsze wyniki.
# Jeżeli chodzi o funkcji straty to dla wszystkich modeli użyto funkcji categorical_crossentropy. Jednak gdyby ją
# zmienić na inną program docelowo nie działałby/spadłby jego wynik, z uwagi na to że funkcja straty działa
# w charakterystyczny sposób dla danego problemu.

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, batch_size=256)
# f) Zmiany rozmiaru partii batch_size możemy dopisać w funkcji fit. Zmiana rozmiaru partii
# wpływa na to jak często model aktualizuje swoje wagi. Bath_size wpływa na stablinosc i dokladnosc
# nauki mowi o tym co ile probek ma zmienic wagi. Mniejsza ilosc zwieksza ryzyko ale tez przyspiesza
# nauke, z kolei wieksza ilosc minimalizuje ryzyko skokow jest stabilniejsze ale moze byc zbyt ogolne i powolne.

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_accuracy*100:.2f}%")
# Plot the learning curve
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid(True, linestyle='--', color='grey')
plt.legend()
plt.tight_layout()
plt.savefig('learning_curve256.png')
# plt.show()
# Save the model
# model.save('iris_model.h5')
model.save('iris_model.keras')
# Plot and save the model architecture
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

#g) Model osiąga najwyższą dokładność (accuracy) w końcowych epokach, gdzie wartości accuracy 
# dla zbioru treningowego i walidacyjnego są bliskie.
# Z kolei strata (loss) konsekwentnie maleje, co sugeruje skuteczne uczenie. Najlepsze wyniki
# osiągnięto w ostatnich epokach, co wskazuje na stabilność modelu. 

#h) Ten kod jest używany do przetwarzania, 
# trenowania i oceny modelu sieci neuronowej na zbiorze danych Iris