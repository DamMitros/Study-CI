import matplotlib.pyplot as plt
import random
import time
from aco import AntColony
plt.style.use("dark_background")

filenames = [ 
    "aco-tsp_ant_count_50", "aco-tsp_ant_count_300", "aco-tsp_ant_count_500",
    "aco-tsp_alpha_0.5", "aco-tsp_alpha_1.0", "aco-tsp_alpha_2.0",
    "aco-tsp_beta_1.2", "aco-tsp_beta_2.7", "aco-tsp_beta_5.0",
    "aco-tsp_evaporation_0.40", "aco-tsp_evaporation_0.60", "aco-tsp_evaporation_0.80",
    "aco-tsp_constant_250", "aco-tsp_constant_1000", "aco-tsp_constant_1500",]
filename = "aco-tsp.png"
params = {
    #ant_count
    # "ant_count": 50, "alpha": 0.5, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 300, "alpha": 0.5, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 500, "alpha": 0.5, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300

    # alpha
    # "ant_count": 300, "alpha": 0.5, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 300, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 300, "alpha": 2.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300

    #beta
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 2.7, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 5.0, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300

    # pheromone_evaporation_rate
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.60, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.80, "pheromone_constant": 1000.0, "iterations": 300

    # pheromone_constant
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 250.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1000.0, "iterations": 300
    # "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.40, "pheromone_constant": 1500.0, "iterations": 300

    "ant_count": 50, "alpha": 1.0, "beta": 1.2, "pheromone_evaporation_rate": 0.60, "pheromone_constant": 1000.0, "iterations": 300
}

COORDS = (
    (20, 52),
    (43, 50),
    (20, 84),
    (70, 65),
    (29, 90),
    (87, 83),
    (73, 23),
    (95, 50),
    (60, 20),
    (10, 10)
)

def random_coord():
    r = random.randint(0, len(COORDS))
    return r

def plot_nodes(w=12, h=8):
    for x, y in COORDS:
        plt.plot(x, y, "g.", markersize=15)
    fig = plt.gcf()
    fig.set_size_inches([w, h])

def plot_all_edges():
    paths = ((a, b) for a in COORDS for b in COORDS)
    for a, b in paths:
        plt.plot((a[0], b[0]), (a[1], b[1]))

plot_nodes()

start_time = time.time()
colony = AntColony(COORDS, **params)

optimal_nodes = colony.get_path()

for i in range(len(optimal_nodes) - 1):
    plt.plot(
        (optimal_nodes[i][0], optimal_nodes[i + 1][0]),
        (optimal_nodes[i][1], optimal_nodes[i + 1][1]),
    )

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.2f} seconds")
plt.savefig(filename, dpi=300, bbox_inches="tight")

# Przed dodaniem nowych punktów, cały proces trwał około 1min (52,65s). (aco-tsp.png) (7 punktów) (231.55104875389213 - ścieżka)
# Po dodaniu nowych punktów, czas wzrósł lekko ponad 1min (71,29s). (aco-tsp(1).png) (12 punktów) (422.2368412134523 - ścieżka)
# Wydaje się, że dodanie nowych punktów nie miało dużego wpływu na czas wykonania algorytmu. Do kolejych 
# testów brane pod uwage będzie 10 punktów. Jednak w przypadku 12 punktów, rój odnalazł niezbyt wydajną
# ścieżkę, co może sugerować, że algorytm nie był w stanie znaleźć optymalnej ścieżki.

# Ilość 'mrówek' (ant_count):
# - 50 mrówek - 336.3889034340639, 10.41s -- wydajne rozwiązanie, dobrze rozwiązane (mała liczba mrówek w koloni)
# - 300 mrówek - 370.88923320758255, 61,79s -- pierwotne rozwiązanie, najgorsze rozwiązanie  
# - 500 mrówek - 356.2023518454058, 103.53s -- długie wykonanie, rozwiązanie gorsze 50 mrówek
# Takie wyniki sugerują, że większa liczba mrówek niekoniecznie prowadzi do lepszego rozwiązania. Wydaje się,
# że wieksza liczba mrówek zaczyna się gubić w poszukiwaniach, co prowadzi do gorszych wyników. (kwestia alphy?)
# NOTE: Drugie wykonanie z 300 mrówkami taki sam wynik jak pierwsze wykonanie z 50 mrówkami. Kwestia losowości?

# Alpha (zaufanie stadne):
# - 0.5 - 336.3889034340639, 61,45s 
# - 1.0 - 323.6545432852203, 61,70s -- najlepsze rozwiązanie
# - 2.0 - 358.4129613134493, 61,96s -- nadmierne zaufanie, gorsze rozwiązanie

# Kluczowe dla czasu jest rozmiar stada, alpha nie wpływa znacząco na czas wykonania. 
# Dla kolejnych elementów ant_count będzie 50, alpha 1.0

# Beta (zaufanie indywidualne):
# - 1.2 - 370.88923320758255, 10,69s
# - 2.7 - 422.9846343985291, 10.48s  // 336.8903364868396, 10.40s // 378.73126854861437, 10.37s
# - 5.0 - 372.54605669721036, 10.46s 

# Beta nie wpływa znacząco na czas wykonania, ale zmienia znacznie rozwiązanie za każdym wykonaniem. Im 
# większa beta tym większa niestabilność rozwiązania.

# Pheromone evaporation rate (czas zapomnienia/parowania feromonów):
# - 0.40 - 372.54605669721036, 10.42s
# - 0.60 - 336.3889034340639, 10.61s 
# - 0.80 - 378.73126854861437, 10.67s

# Pheromone constant (siła feromonów):
# - 250.0 - 378.73126854861437, 10.23s
# - 1000.0 - 336.3889034340639, 10.33s
# - 1500.0 - 372.54605669721036, 10.41s

# Czas wykonania nie zmienia się znacząco zmieniając parametry, poza ilością mrówek. Pozostałe parametry
# minimalnie zwiększa czas wykonania w zależności od wielkości ale nie są to czynniki decydujące.
# Ilość mrówek jest kluczowym czynnikiem decydującym o czasie wykonania algorytmu.

# Najbardziej optymalne rozwiązanie to:
# - 50 mrówek, alpha 1.0, beta 1.2, pheromone_evaporation_rate 0.60, pheromone_constant 1000.0
# 336.3889034340639, 356.2023518454058, 336.3889034340639, 336.8903364868396  <11s