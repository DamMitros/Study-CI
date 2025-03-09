import pandas as pd
from sklearn import datasets
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='FlowerType')
print(X.head())

pca_iris = PCA(n_components=2).fit(iris.data) 
print(pca_iris)

print("Wariancja wyjaśniona przez kolejne składowe PCA:") 
explained_variance = pca_iris.explained_variance_ratio_ 
print(explained_variance)

total_variance = sum(explained_variance[:2]) 
print(f"Łączna zachowana wariancja dla 2 wymiarów: {total_variance:.2%}")

# Wariancje wykazała, że dla dwóch pierwszych składowych zachowujemy 97.77% wariancji, co jest większe niż 95%.
# Dlatego można usunąć dwie kolumny, tak aby zachować minimum 95% wariancji.

print("Macierz komponentów PCA dla orginalnych danych:")
print(pca_iris.components_)

# DataFrame na bazie dwóch głównych składowych
transformed_iris = pca_iris.transform(iris.data)
pca=pd.DataFrame(transformed_iris, columns=['PCA1', 'PCA2'])
pca['FlowerType'] = y

plt.figure(figsize=(8, 6))
scatter=plt.scatter(pca['PCA1'], pca['PCA2'], c=pca['FlowerType'], cmap='viridis')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('PCA zbioru danych iris')
plt.legend(handles=scatter.legend_elements()[0], labels=['Setosa', 'Versicolor', 'Virginica'])
plt.savefig('PCA.png')
