import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

df=pd.read_csv('iris1.csv')

#a)
(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292586)

#b)
k_values=[3,5,11]
for k in k_values:
  knn=KNeighborsClassifier(n_neighbors=k)
  knn.fit(train_set[:,0:4], train_set[:,4])
  accuracy=knn.score(test_set[:,0:4], test_set[:,4])
  print(f'KNN, k={k}, accuracy={accuracy}')

nb=GaussianNB()
nb.fit(train_set[:,0:4], train_set[:,4])
accuracy=nb.score(test_set[:,0:4], test_set[:,4])
print(f'Naive Bayes, accuracy={accuracy}')

# W kontekście dokładności klasyfikatorów, ten sam wynik został osiągniety dla wszystkich 
# wartości k w k-NN oraz DD. W przypadku klasyfikatora Naive Bayes, wynik jest nieco niższy.