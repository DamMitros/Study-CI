import pandas as pd 
from sklearn.model_selection import train_test_split

df=pd.read_csv('iris1.csv')

(train_set, test_set) = train_test_split(df.values, train_size=0.7, random_state=292586)
# train_test_split zwraca dwie tablice, pierwsza to train_set, a druga to test_set
# train_size=0.7 oznacza, że 70% danych trafi do train_set, a 30% do test_set

def classify_iris(sl, sw, pl, pw):
  if pl < 2:
    return('Setosa')
  elif pl >= 5:
    return('Virginica')
  else:
    return('Versicolor')
  
good_predictions = 0
len = test_set.shape[0]

for i in range(len):
  if classify_iris(test_set[i][0], test_set[i][1], test_set[i][2], test_set[i][3]) == test_set[i][4]: 
    # uzupełnienie warunków, aby funkcja działała następnie sprawdzała czy wynik predykcji jest równy wartości w kolumnie 4
    good_predictions += 1

print(good_predictions)
print(good_predictions/len*100, "%") 

# Przed korektą:
# 16/45 poprawnych predykcji, 35,56% poprawności 
# Po korekcie:
# 41/45 poprawnych predykcji, 91,11% poprawności

#posortowanie train_set po kolumnie 4 w celu poprawienia dokładności predykcji
train_set = train_set[train_set[:,4].argsort()]
# print(train_set)

#pierwszym elementem na który zwróciłem uwagę była długość płatka (petal lenght), gdzie na oko zauwazyłem, że 
#Virginica ma długość płatka powyżej 5 (z pojedyńczymi wyjątkami), a Setosa poniżej 2. Po zapisaniu tego jako 
#warunek wynik drastycznie wzrósł do 91,11% poprawności. Wystarczyło to aby osiągnąć wynik powyżej 90% poprawności.
