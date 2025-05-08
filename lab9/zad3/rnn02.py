import numpy as np
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN


def create_RNN(hidden_units, dense_units, input_shape, activation):
    model = Sequential() # stworzenie modelu sekwencyjnego RNN
    model.add(SimpleRNN(hidden_units, input_shape=input_shape, 
                        activation=activation[0])) # dodanie warstwy z daną liczbą neuronów i kształtem wejściowym 
    model.add(Dense(units=dense_units, activation=activation[1])) # dodanie warstwy gęstej z daną liczbą neuronów i funkcją aktywacji
    model.compile(loss='mean_squared_error', optimizer='adam') # kompilacja modelu z funkcją straty i optymalizatorem
    return model

demo_model = create_RNN(2, 1, (3,1), activation=['linear', 'linear']) 

wx = demo_model.get_weights()[0] # wagi wejściowe
wh = demo_model.get_weights()[1] # wagi ukryte
bh = demo_model.get_weights()[2] # bias ukryty
wy = demo_model.get_weights()[3] # wagi wyjściowe
by = demo_model.get_weights()[4] # bias wyjściowy

print('wx = ', wx, ' wh = ', wh, ' bh = ', bh, ' wy =', wy, 'by = ', by)

x = np.array([1, 2, 3])
# Reshape the input to the required sample_size x time_steps x features 
x_input = np.reshape(x,(1, 3, 1))
y_pred_model = demo_model.predict(x_input) 
# przewidywanie wyjścia modelu na podstawie danych wejściowych

m = 2
h0 = np.zeros(m)
h1 = np.dot(x[0], wx) + h0 + bh
h2 = np.dot(x[1], wx) + np.dot(h1,wh) + bh
h3 = np.dot(x[2], wx) + np.dot(h2,wh) + bh
o3 = np.dot(h3, wy) + by
# ręczne obliczenie wyjścia na podstawie danych wejściowych i wag

print('h1 = ', h1,'h2 = ', h2,'h3 = ', h3)

print("Prediction from network ", y_pred_model) # [[1.7165807]]
print("Prediction from our computation ", o3)   # [[1.71658082]]

# 1. Tworzymy model rnn (2 warstwy i kompilator)
# 2. Zbieramy wagi i bias z modelu
# 3. Predyktujemy wyniki
# 4. Obliczamy reczeni predykcje
# 5. Porównujemy wynik reczny a modelu