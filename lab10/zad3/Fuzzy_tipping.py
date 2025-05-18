import simpful as sf, numpy as np
import matplotlib.pyplot as plt

FS = sf.FuzzySystem()

# Zdefiniowane kategorie jakości jedzenia i serwisu ( poor (0-5), good (5-10), excellent (10))
S_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="poor")
S_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="good")
S_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="excellent")
FS.add_linguistic_variable("food", sf.LinguisticVariable([S_1, S_2, S_3], universe_of_discourse=[0, 10]))

T_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=5), term="poor")
T_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=5, c=10), term="good")
T_3 = sf.FuzzySet(function=sf.Triangular_MF(a=5, b=10, c=10), term="excellent")
FS.add_linguistic_variable("service", sf.LinguisticVariable([T_1, T_2, T_3], universe_of_discourse=[0, 10]))

# Zdefiniowane kategorie napiwków (low (0-15), medium (15-30), high (30))
U_1 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=0, c=15), term="low")
U_2 = sf.FuzzySet(function=sf.Triangular_MF(a=0, b=15, c=30), term="medium")
U_3 = sf.FuzzySet(function=sf.Triangular_MF(a=15, b=30, c=30), term="high")
FS.add_linguistic_variable("tip", sf.LinguisticVariable([U_1, U_2, U_3], universe_of_discourse=[0, 30]))

# Fuzzy rules
R1 = "IF (food IS poor) OR (service IS poor) THEN (tip IS low)"
R2 = "IF (service IS good) THEN (tip IS medium)"
R3 = "IF (food IS excellent) OR (service IS excellent) THEN (tip IS high)"

FS.add_rules([R1, R2, R3])

FS.set_crisp_output_value("low", 0)
FS.set_crisp_output_value("medium", 10)
FS.set_crisp_output_value("high", 20)

test_inputs = [
    (2, 2),   # Słaba jakość jedzenia, słaba obsługa
    (8, 2),   # Dobra jakość jedzenia, słaba obsługa
    (2, 8),   # Słaba jakość jedzenia, dobra obsługa
    (5, 5),   # OK jakość jedzenia, OK obsługa
    (9, 9)    # Wspania jakość jedzenia, wspaniała obsługa
]

print("Testowanie systemu sterowania rozmytego dla napiwków:")
print("--------------------------------------------")
print(f"{'Jakość jedzenia':>16} | {'Jakość obsługi':>15} | {'Napiwek':>8}")
print("--------------------------------------------")

for food_val, service_val in test_inputs:
  FS.set_variable("food", food_val)
  FS.set_variable("service", service_val)
  tip = FS.Mamdani_inference(["tip"])
  print(f"{food_val:16.2f} | {service_val:15.2f} | {tip['tip']:8.2f}%")

print("--------------------------------------------")

def calculate_tip(food_val, service_val):
  FS.set_variable("food", food_val)
  FS.set_variable("service", service_val)
  tip = FS.Mamdani_inference(["tip"])
  return tip['tip']

food_range = np.linspace(0, 10, 20)
service_range = np.linspace(0, 10, 20)
food_grid, service_grid = np.meshgrid(food_range, service_range)
tip_grid = np.zeros_like(food_grid)

for i in range(len(food_range)):
  for j in range(len(service_range)):
    tip_grid[j, i] = calculate_tip(food_range[i], service_range[j])

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(food_grid, service_grid, tip_grid, cmap='viridis', edgecolor='none')
ax.set_xlabel('Food Quality')
ax.set_ylabel('Service Quality')
ax.set_zlabel('Tip Percentage')
ax.set_title('Fuzzy Logic Controller for Tipping')
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
plt.savefig('Fuzzy_Tipping_Surface.png')