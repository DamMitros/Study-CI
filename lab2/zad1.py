import pandas as pd

df = pd.read_csv('iris_with_errors.csv')
# a)
empty_data = df.isna().sum() + (df == ' ').sum() + (df == '-').sum()
print(empty_data)
# b):

# b) Sprawdź czy wszystkie dane numeryczne są z zakresu (0; 15). Dane spoza zakresu muszą być poprawione. Możesz
# tutaj użyć metody: za błędne dane podstaw średnią (lub medianę) z danej kolumny
for value in df.values:
  for i in range(4):
    if not 0 < value[i] < 15:
      df.iloc[value[0], i] = df.iloc[:, i].mean()
