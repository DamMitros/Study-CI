import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('iris1.csv')

(X_train, X_test, y_train, y_test)=train_test_split(df.iloc[:,:4], df.iloc[:,4], train_size=0.7, random_state=292586)

# b)
clf=DecisionTreeClassifier()

# c)
clf.fit(X_train, y_train)

# d)
print("Drzewko decyzyjne:")
print(export_text(clf, feature_names=df.columns[:4].tolist()))

# e)
accuracy=clf.score(X_test, y_test)
print("Dokładność modelu:", accuracy)

# f)
y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Macierz błędów: \n", cm)

# dodatkowo macierz błedów w formie graficznej
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=df['variety'].unique(), yticklabels=df['variety'].unique())
plt.xlabel("Predykcja")
plt.ylabel("Rzeczywista klasa")
plt.title("Macierz błędów")
plt.savefig("confusion_matrix.png")

# Wychodzi na to, że mamy remis z AI, oba modele osiągneły ten sam wynik 91,11% poprawności.