from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 


df=pd.read_csv('diabetes 1.csv')

X=df.iloc[:,:8]
# y=df.iloc[:,8]
y = df.iloc[:,8].map({'tested_negative': 0, 'tested_positive': 1})

(X_train, X_test, y_train, y_test)=train_test_split(X, y, train_size=0.7, random_state=1081944)

scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

model1 = MLPClassifier(hidden_layer_sizes=(6,3), activation='relu', max_iter=500, random_state=2025)
model1.fit(X_train_scaled, y_train)

accuracy1= model1.score(X_test_scaled, y_test)
print("Dokładność modelu z 6 i 3 neuronami:", accuracy1)

y_pred = model1.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
print("Macierz błędów: \n", cm)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz błędów")
plt.savefig("confusion_matrix.png")

# W przypadku modelu z 6 i 3 neuronami, dokładność wynosi 75,75%. W przypadku macierzy błędów,
# widać, że model często myli się w klasyfikacji osób z cukrzycą.

model2 = MLPClassifier(hidden_layer_sizes=(4,2), activation='tanh', max_iter=2000, random_state=2025)
model2.fit(X_train_scaled, y_train)

accuracy2= model2.score(X_test_scaled, y_test)
print("Dokładność modelu z 4 i 2 neuronami:", accuracy2)

model3 = MLPClassifier(hidden_layer_sizes=(6,5,4), activation='logistic', max_iter=2000, random_state=2025)
model3.fit(X_train_scaled, y_train)

accuracy3= model3.score(X_test_scaled, y_test)
print("Dokładność modelu z 6,5 i 4 neuronami:", accuracy3)

# Sieć neuronowa poradziła sobie gorzej niż poprzednie klasyfikatore
#  w porównaniu do tych z poprzednich laboratoriów.

# Liczbowo jest więcej FN, co jest zdecydowanie gorsze niż FP, ponieważ
# w przypadku cukrzycy, lepiej jest zdiagnozować ją, niż nie zdiagnozować.
