import sklearn.datasets as datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# a)
iris = datasets.load_iris()
X = iris.data
y = iris.target

(X_train, X_test, y_train, y_test) = train_test_split(X, y, train_size=0.7, random_state=8080)

# b)

# W tym przypadku rozwiązania pobranie danych z biblioteki powoduję, że
# train_labels i test_labels są już zdefiniowane jako liczby, a nie napisy.
# Gdyby pobrać dane z pliku CSV, wtedy należałoby je odpowiednio przekonwertować.

for i, label in enumerate(iris.target_names):
  print(f"{i}: {label}")

# 0: setosa
# 1: versicolor
# 2: virginica

# c)
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

# d)
# Model 1 -> 4 wejścia, 2 neurony w warstwie ukrytej -> 3 wyjścia 
model1 = MLPClassifier(hidden_layer_sizes=(2,), max_iter=4000, random_state=42)
model1.fit(X_train_scaled, y_train)

# e)
accuracy1= model1.score(X_test_scaled, y_test)
print("Dokładność modelu z 2 neutronami:", accuracy1)

# f) 
# Model 2 -> 4 wejścia, 3 neurony w warstwie ukrytej -> 3 wyjścia
model2 = MLPClassifier(hidden_layer_sizes=(3,), max_iter=4000, random_state=42)
model2.fit(X_train_scaled, y_train)

accuracy2= model2.score(X_test_scaled, y_test)
print("Dokładność modelu z 3 neutronami:", accuracy2)

# g)
# Model 3 -> 4 wejścia -> 3 neurony -> 3 neurony -> 3 wyjścia
model3 = MLPClassifier(hidden_layer_sizes=(3, 3), max_iter=4000, random_state=42)
model3.fit(X_train_scaled, y_train)

accuracy3= model3.score(X_test_scaled, y_test)
print("Dokładność modelu z dwoma wartwami po 3 neutrony:", accuracy3)

# Najlepiej wypada model2, w którym 3 neurony w warstwie ukrytej osiągnęły dokładność 97,78%. Aktualnie przy 
# zwiększonej iteracji model1 osiąga ten sam wynik, jednak przy niższych wartościach osiągał gorszy efekt.