import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def data_plot(df, ax, title):
  for species in df['variety'].unique():
    data = df[df['variety'] == species]
    ax.scatter(data['sepal.length'], data['sepal.width'], label=species)
  ax.set_title(title)
  ax.set_xlabel('sepal.length')
  ax.set_ylabel('sepal.width')
  ax.legend()

df=pd.read_csv('iris1.csv')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

# Dane oryginalne
data_plot(df, ax1, 'Dane oryginalne')

# Znormalizowane min-max
scaler_minmax=MinMaxScaler()
X_minmax=scaler_minmax.fit_transform(df[['sepal.length', 'sepal.width']])
df_minmax=pd.DataFrame(X_minmax, columns=['sepal.length', 'sepal.width'])
df_minmax['variety']=df['variety']
data_plot(df_minmax, ax2, 'Znormalizowane min-max')

# Zeskalowane z-scorem(standaryzacja)
scaler_zscore=StandardScaler()
X_zscore=scaler_zscore.fit_transform(df[['sepal.length', 'sepal.width']])
df_zscore=pd.DataFrame(X_zscore, columns=['sepal.length', 'sepal.width'])
df_zscore['variety']=df['variety']
data_plot(df_zscore, ax3, 'Zeskalowane z-scorem')

plt.savefig('iris.png')

# Na podstawie wykresów można stwierdzić, że na pierwszy rzut oka nie ma dużych różnic między metodami skalowania.
# Wszystkie trzy wykresy wyglądają bardzo podobnie. Jednak po bliższym przyjrzeniu się można zauważyć, że dane
# orginalne mają większe wartości od znormalizowanych min-max i zeskalowanych z-scorem. Wartości znormalizowane
# min-max są w przedziale od 0 do 1, a zeskalowane z-scorem są zeskalowane tak, że średnia wynosi 0, a odchylenie
# standardowe wynosi 1. Wartości zeskalowane z-scorem są zatem bardziej skupione wokół zera.