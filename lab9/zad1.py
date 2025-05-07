import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv('titanic.csv', sep=',', encoding='utf-8')
required_columns = ["Class", "Sex", "Age", "Survived"]
df = df[required_columns]

df_encoded = pd.get_dummies(df)
print("Sample of one-hot encoded data:")
print(df_encoded.head())

frequent_itemsets = apriori(df_encoded, min_support=0.005, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.8)
rules = rules.sort_values(by='confidence', ascending=False)

interesting_rules = rules[
  # rules['consequents'].apply(lambda x: any(item.startswith('Survived_') for item in x))
  rules['antecedents'].apply(lambda x: any(item.startswith('Survived_') for item in x))
]

# antecedents = zestaw cech, które muszą być spełnione
# consequents = zestaw cech, które są przewidywane
# support = odsetek bazy danych, w której występują dane cechy
# confidence = prawdopodobieństwo, że (antecedents) występują razem z (consequents)
# lift = stosunek (np. przetrwania) w danej grupie do ogólnego przetrwania
print(interesting_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())
print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].to_string())

plt.figure(figsize=(10, 6))
plt.scatter(rules['support'], rules['confidence'], c=rules['lift'], cmap='viridis', alpha=0.7)
plt.colorbar(label='Lift')
plt.xlabel('Support')
plt.ylabel('Confidence')
plt.title('Association Rules: Support vs. Confidence (Color-coded by Lift)')
plt.grid(True)
plt.savefig('association_rules_plot.png')

# Interpretacja wykresu:
# - Każdy punkt na wykresie reprezentuje jedną regułę asocjacyjną.
# - Oś X (Support) pokazuje, jak często dana kombinacja elementów (antecedent i consequent) występuje w zbiorze danych.
# - Oś Y (Confidence) pokazuje, jak często consequent występuje, gdy antecedent jest obecny.
# - Kolor punktu (Lift) wskazuje, jak bardzo prawdopodobieństwo wystąpienia consequenta wzrasta, gdy antecedent jest obecny,
#   w porównaniu do ogólnego prawdopodobieństwa wystąpienia consequenta.
#   - Lift > 1 oznacza, że antecedent i consequent występują razem częściej niż oczekiwano by losowo (pozytywna korelacja).
#   - Lift < 1 oznacza, że występują razem rzadziej niż oczekiwano (negatywna korelacja).
#   - Lift = 1 oznacza brak korelacji.