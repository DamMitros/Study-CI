# Przefiltrowanie reguł, aby znaleźć te związane z przeżyciem w consequentach

```
#                             antecedents                consequents   support  confidence      lift
# 22               (Class_2nd, Age_Child)             (Survived_Yes)  0.010904    1.000000  3.095640
# 11              (Class_1st, Sex_Female)             (Survived_Yes)  0.064062    0.972414  3.010243
# 45   (Age_Adult, Class_1st, Sex_Female)             (Survived_Yes)  0.063607    0.972222  3.009650
# 47              (Class_1st, Sex_Female)  (Age_Adult, Survived_Yes)  0.063607    0.965517  3.249394
# 58     (Age_Adult, Class_2nd, Sex_Male)              (Survived_No)  0.069968    0.916667  1.354083
# 17              (Class_2nd, Sex_Female)             (Survived_Yes)  0.042254    0.877358  2.715986
# 72  (Class_Crew, Sex_Female, Age_Adult)             (Survived_Yes)  0.009087    0.869565  2.691861
# 31             (Class_Crew, Sex_Female)             (Survived_Yes)  0.009087    0.869565  2.691861
# 73             (Class_Crew, Sex_Female)  (Age_Adult, Survived_Yes)  0.009087    0.869565  2.926473
# 20                (Class_2nd, Sex_Male)              (Survived_No)  0.069968    0.860335  1.270871
# 61                (Class_2nd, Sex_Male)   (Age_Adult, Survived_No)  0.069968    0.860335  1.316827
# 54   (Age_Adult, Class_2nd, Sex_Female)             (Survived_Yes)  0.036347    0.860215  2.662916
# 68     (Age_Adult, Class_3rd, Sex_Male)              (Survived_No)  0.175829    0.837662  1.237379
# 26                (Class_3rd, Sex_Male)              (Survived_No)  0.191731    0.827451  1.222295
```

# Podsumowanie:
- Dzieci w 2 klasie miały 100% szans na przeżycie, ale był to tylko 1% wszystkich pasażerów.
- Kobiety w 1 klasie miały 97% szans na przeżycie, ale stanowiły tylko 6% wszystkich pasażerów.
- Mężczyźni w 2 klasie mieli podobne szanse na nieprzeżycie (92%, 7% pasażerów) jak kobiety w 2 klasie na przeżycie (87%, 7% pasażerów).
- Kobiety z załogi i 2 klasy miały około 87% (5% pasażerów) na przeżycie (kobiety z 1 klasy miały 97% szans na przeżycie, 6% pasażerów).
- Mężczyźni w 3 klasie mieli większe szanse na przeżycie (82%, 19% pasażerów) niż mężczyźni w 2 klasie (86%, 7% pasażerów).

# Najciekawsze korelacje ze wszystkich reguł:

## Podział na płcie
### Wpływ klasy na mężczyznę:
Jako mężczyzna najłatwiej było przeżyć w 1 klasie (z około 8% wszystkich pasażerów - 2.5% przeżyło => 33% szansy).
```
# 12                   (Sex_Male, Class_1st)                (Age_Adult)  0.079509    0.972222  1.022878
# 51     (Sex_Male, Survived_Yes, Class_1st)                (Age_Adult)  0.025897    0.919355  0.967256
```
Kolejni była załoga (około 39% wszystkich pasażerów - 8.7% przeżyło => 22,3% szansy).
```
# 33                 (Class_Crew, Age_Adult)                 (Sex_Male)  0.391640    0.974011  1.238474
# 78   (Class_Crew, Survived_Yes, Age_Adult)                 (Sex_Male)  0.087233    0.905660  1.151565
```
W klasie 2 mężczyźni (z około 7% wszystkich pasażerów - 0.6% przeżyło => 8,3% szansy).
```
# 18                   (Class_2nd, Sex_Male)                (Age_Adult)  0.076329    0.938547  0.987449
# 57      (Class_2nd, Sex_Male, Survived_No)                (Age_Adult)  0.069968    1.000000  1.052103
```
W 3 klasie (z około 21% wszystkich pasażerów - 3.4% przeżyło => 16,3% szansy).
```
# 25                   (Sex_Male, Class_3rd)                (Age_Adult)  0.209905    0.905882  0.953082
# 67        (Sex_Male, Class_3rd, Age_Adult)              (Survived_No)  0.175829    0.837662  1.237379
```

### Wpływ klasy na kobietę:
Jako kobieta najłatwiej było przeżyć w 1 klasie (z około 6% wszystkich pasażerów - 97% przeżyło).
```
# 10                 (Sex_Female, Class_1st)                (Age_Adult)  0.065425    0.993103  1.044847
# 45   (Age_Adult, Class_1st, Sex_Female)             (Survived_Yes)  0.063607    0.972222  3.009650
```
Bycie kobietą w drugiej klasie dawało pewność przeżycia (4% pasażerów - 100%! przeżyło).
```
# 17                 (Class_2nd, Sex_Female)             (Survived_Yes)  0.042254    0.877358  2.715986
# 16                 (Class_2nd, Sex_Female)                (Age_Adult)  0.042254    0.877358  0.923072
```
Kobiety w 3 klasie (7.5% pasażerów - 3.5% przeżyło => 47%).
```
# 23                 (Class_3rd, Sex_Female)                (Age_Adult)  0.074966    0.841837  0.885699
# 64   (Class_3rd, Sex_Female, Survived_Yes)                (Age_Adult)  0.034530    0.844444  0.888443
```
Kobiety w załodze (około 1% pasażerów - 87% przeżyło).
```
# 31                (Class_Crew, Sex_Female)             (Survived_Yes)  0.009087    0.869565  2.691861
# 30                (Class_Crew, Sex_Female)                (Age_Adult)  0.010450    1.000000  1.052103
```

## Dzieci:
- Dzieci w 1 klasie - brak danych.
- Dzieci w 2 klasie (około 1% pasażerów - 100% przeżyło).
```
# 56   (Class_2nd, Sex_Female, Age_Child)             (Survived_Yes)  0.005906    1.000000  3.095640
```
- Dzieci w 3 klasie (około 2% pasażerów - 0% przeżyło).
```
# 29                (Age_Child, Survived_No)                (Class_3rd)  0.023626    1.000000  3.117564
# 70      (Sex_Male, Age_Child, Survived_No)                (Class_3rd)  0.015902    1.000000  3.117564
# 65    (Survived_No, Age_Child, Sex_Female)                (Class_3rd)  0.007724    1.000000  3.117564
```
- Brak dzieci w załodze.

## Stosunek przeżytych do wszystkich pasażerów (z 95% dorosłych przeżyło około 30% pasażerów):
```
# 6                               (Sex_Male)                (Age_Adult)  0.757383    0.963027  1.013204
# 5                             (Sex_Female)                (Age_Adult)  0.193094    0.904255  0.951370
# 9                           (Survived_Yes)                (Age_Adult)  0.297138    0.919831  0.967757
```
Jeżeli chodzi o dzieci (to z 3% przeżyło około 1% pasażerów):
- Brak w statystykach lekko ponad 1% pasażerów (dzieci z 1 klasy?).

## Przeżycie kobiet i mężczyzn (przeżyło około 30% pasażerów w tym 14.4% kobiet i 15.5% mężczyzn):
```
# 40              (Survived_Yes, Sex_Female)                (Age_Adult)  0.143571    0.918605  0.966467
# 44                (Sex_Male, Survived_Yes)                (Age_Adult)  0.153567    0.920981  0.968967
```

Ze wszystkich analiz wynika, że priorytetem ocalenia były kobiety i dzieci z 1 i 2 klasy oraz załoga.
Mężczyźni byli główną grupą pasażerów, ale względem kobiet ocaleli w większej liczbie.
Dzieci w 2 klasie miały 100% szans na przeżycie, kiedy z 3 klasy nie przeżyło ani jedno.
Kobiety w 1 i 2 klasie ocalały prawie wszystkie, a w 3 klasie tylko 47%. Mężczyźni mieli
inny los i znacznie mniej przeżyło w 1 klasie, z kolei w 2 i 3 statystyka wygląda dramatycznie.
Nieznany jest los dzieci w 1 klasie!