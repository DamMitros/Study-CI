import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Biblioteka do wizualizacji danych
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Wczytanie danych
file_path = "iris1.csv"  # Upewnij się, że plik jest w tym samym katalogu
iris = pd.read_csv(file_path)

# Wyświetlenie podstawowych statystyk
print("Podstawowe statystyki oryginalnych danych:")
print(iris[['sepal.length', 'sepal.width']].describe())

# Normalizacja Min-Max
df_minmax = iris.copy()
scaler_minmax = MinMaxScaler()
df_minmax[['sepal.length', 'sepal.width']] = scaler_minmax.fit_transform(df_minmax[['sepal.length', 'sepal.width']])

# Standaryzacja Z-score
df_zscore = iris.copy()
scaler_zscore = StandardScaler()
df_zscore[['sepal.length', 'sepal.width']] = scaler_zscore.fit_transform(df_zscore[['sepal.length', 'sepal.width']])

# Tworzenie wykresów
def plot_iris(df, title, ax):
    sns.scatterplot(x='sepal.length', y='sepal.width', hue='variety', data=df, ax=ax)
    ax.set_title(title)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
plot_iris(iris, "Oryginalne dane", axes[0])
plot_iris(df_minmax, "Znormalizowane Min-Max", axes[1])
plot_iris(df_zscore, "Zeskalowane Z-score", axes[2])

plt.tight_layout()
# plt.show()
plt.savefig('irisGPT.png')
