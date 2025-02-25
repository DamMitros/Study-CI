import numpy as np
import matplotlib.pyplot as plt
import random

# Stałe
g = 9.81  # Przyspieszenie grawitacyjne [m/s^2]
v0 = 50    # Prędkość początkowa [m/s]
h = 100    # Wysokość początkowa [m]

# Losowanie odległości celu
target_distance = random.randint(50, 340)
print(f"Cel znajduje się w odległości: {target_distance} metrów.")

def calculate_range(angle):
    """Oblicza zasięg pocisku na podstawie kąta strzału."""
    angle_rad = np.radians(angle)
    v0x = v0 * np.cos(angle_rad)
    v0y = v0 * np.sin(angle_rad)
    
    # Czas lotu na podstawie równań ruchu
    t_flight = (v0y + np.sqrt(v0y**2 + 2 * g * h)) / g
    
    # Zasięg poziomy
    return v0x * t_flight

attempts = 0
while True:
    try:
        angle = float(input("Podaj kąt strzału (w stopniach): "))
        attempts += 1
        shot_distance = calculate_range(angle)
        print(f"Twój strzał trafił w odległość: {shot_distance:.2f} metrów.")
        
        if target_distance - 5 <= shot_distance <= target_distance + 5:
            print(f"Cel trafiony! Liczba prób: {attempts}")
            break
        else:
            print("Chybiony! Spróbuj ponownie.")
    except ValueError:
        print("Błąd: Wprowadź poprawną wartość kąta!")

# Rysowanie trajektorii, jeśli cel trafiony
def plot_trajectory(angle):
    angle_rad = np.radians(angle)
    v0x = v0 * np.cos(angle_rad)
    v0y = v0 * np.sin(angle_rad)
    
    t_flight = (v0y + np.sqrt(v0y**2 + 2 * g * h)) / g
    t = np.linspace(0, t_flight, num=100)
    
    x = v0x * t
    y = h + v0y * t - 0.5 * g * t**2
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, 'b', label="Trajektoria pocisku")
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(target_distance, color='red', linestyle='dashed', label="Cel")
    plt.xlabel("Odległość [m]")
    plt.ylabel("Wysokość [m]")
    plt.title("Trajektoria pocisku Warwolf")
    plt.legend()
    plt.grid()
    plt.savefig("trajektoriaGPT.png")
    # plt.show()

plot_trajectory(angle)

# Poprosiłem GPT-4 o zrobienie tego zadania i zrobił to w lekko odmienny sposób. Nie stworzył funkcji głównej, użył
# biblioteki numpy do obliczeń oraz zastosował pętlę while zamiast for. Dodatkowo, zastosował funkcję do rysowania
# trajektorii pocisku, gdzie ja korzystałem z metody iteracyjnej w pętli while. Jego wykres ma więcej informacji.