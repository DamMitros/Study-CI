import pygad
import time

item = [
    ("zegar", 100, 7),
    ("obraz-pejzaż", 300, 7),
    ("obraz-portret", 200, 6),
    ("radio", 40, 2),
    ("laptop", 500, 5),
    ("lampka nocna", 70, 6),
    ("srebrne sztućce", 100, 1),
    ("porcelana", 250, 3),
    ("figura z brązu", 300, 10),
    ("skórzana torbka", 280, 3),
    ("odkurzacz", 300, 15)
    ]

max_weight = 25

#definiujemy parametry chromosomu
#geny to liczby: 0 lub 1
gene_space = [0, 1]

#definiujemy funkcję fitness
def fitness_func(ga_instance, solution, solution_idx):  # solution_idx is kept for compatibility
    total_value = 0
    total_weight = 0

    for i in range(len(solution)):
        if solution[i] == 1:
            total_value += item[i][1]
            total_weight += item[i][2]
    if total_weight > max_weight:
        return 0
    if total_value == 1630:
        ga_instance.stop_criteria = ["reach_1630"]

    return total_value

def on_generation(ga_instance):
    if ga_instance.best_solution()[1] == 1630:
        return "reach_1630"
    
fitness_function = fitness_func

#ile chromsomów w populacji
#ile genow ma chromosom
sol_per_pop = 10
num_genes = len(item)

#ile wylaniamy rodzicow do "rozmanazania" (okolo 50% populacji)
#ile pokolen
#ilu rodzicow zachowac (kilka procent)
num_parents_mating = 5
num_generations = 30
keep_parents = 2

#jaki typ selekcji rodzicow?
#sss = steady, rws=roulette, rank = rankingowa, tournament = turniejowa
parent_selection_type = "sss"

#w il =u punktach robic krzyzowanie?
crossover_type = "single_point"

#mutacja ma dzialac na ilu procent genow?
#trzeba pamietac ile genow ma chromosom
mutation_type = "random"
mutation_percent_genes = 10

def run_single_ga():
    start_time = time.time()
    #inicjacja algorytmu z powyzszymi parametrami wpisanymi w atrybuty
    ga_instance = pygad.GA(gene_space=gene_space,
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        on_generation=on_generation)  # Dodano callback funkcję

    #uruchomienie algorytmu
    ga_instance.run()
    end_time=time.time()
    execution_time = end_time-start_time
    
    #podsumowanie: najlepsze znalezione rozwiazanie (chromosom+ocena)
    solution, solution_fitness, solution_idx = ga_instance.best_solution()
     
    return solution, solution_fitness, execution_time, ga_instance  

def test_ga_performance(num_runs=10): 
    optimal_count = 0
    optimal_times = []
    best_solution = None
    best_fitness = 0
    best_ga_instance = None  
    
    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        solution, fitness, exec_time, ga_instance = run_single_ga()
        
        if fitness == 1630:
            optimal_count += 1
            optimal_times.append(exec_time)
            if best_solution is None:
                best_solution = solution
                best_fitness = fitness
                best_ga_instance = ga_instance
        elif fitness > best_fitness:
            best_solution = solution
            best_fitness = fitness
            best_ga_instance = ga_instance
    
    success_rate = (optimal_count / num_runs) * 100
    avg_time = sum(optimal_times) / len(optimal_times) if optimal_times else 0
    
    print(f"\nWyniki na {num_runs} uruchomień:")
    print(f"Procent znalezienia optymalnego rozwiązania (1630 zł): {success_rate}%")
    print(f"Średni czas wykonania dla optymalnych rozwiązań: {avg_time:.4f} s")
    
    return best_solution, best_fitness, best_ga_instance

def display(solution):
    total_value = 0
    total_weight = 0
    for i in range(len(solution)):
        if solution[i] == 1:
            print(item[i][0])
            total_value += item[i][1]
            total_weight += item[i][2]
    print("Total value: ", total_value)
    print("Total weight: ", total_weight)

def main():
    print("Rozwiązanie problemu plecakowego za pomocą algorytmu genetycznego")
    
    print("\nPojedyncze uruchomienie algorytmu:")
    solution, fitness, exec_time, ga_instance = run_single_ga()  
    print(f"Najlepsze rozwiązanie: {solution}")
    print(f"Wartość funkcji fitness: {fitness}")
    display(solution)
    
    ga_instance.plot_fitness().savefig("fitness_plot.png")

    print("\nTestowanie wydajności algorytmu na 10 uruchomieniach...")
    best_solution, best_fitness, best_ga_instance = test_ga_performance(10) 
    
    print("\nNajlepsze znalezione rozwiązanie:")
    display(best_solution)
    # Średni czas rozwiązania - 0.0069s
    # Procent znalezieine rozwiazania 1630 - 80%

    if best_ga_instance: 
        best_ga_instance.plot_fitness().savefig("fitness_plot_best.png")

main()