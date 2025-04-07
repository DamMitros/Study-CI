import pygad
import math

# def 𝑒𝑛𝑑𝑢𝑟𝑎𝑛𝑐𝑒(𝑥, 𝑦, 𝑧, 𝑣, 𝑢, 𝑤)
def endurance(x, y, z, v , u, w):
  return math.exp(-2*(y-math.sin(x))**2) + math.sin(z*u) + math.cos(v*w)

def fitness_func(ga_instance, solution, solution_idx):
  x,y,z,v,u,w = solution
  return endurance(x, y, z, v, u, w)

gene_space = {'low': 0.0 , 'high': 1.0}

sol_per_pop = 10
num_genes = 6
num_parents_mating = 5
num_generations = 50
keep_parents = 5
parent_selection_type = "sss"
crossover_type = "single_point"
mutation_type = "random"
mutation_percent_genes = 20

ga_instance = pygad.GA(gene_space=gene_space,
              num_generations=num_generations,
              num_parents_mating=num_parents_mating,
              fitness_func=fitness_func,
              sol_per_pop=sol_per_pop,
              num_genes=num_genes,
              parent_selection_type=parent_selection_type,
              keep_parents=keep_parents,
              crossover_type=crossover_type,
              mutation_type=mutation_type,
              mutation_percent_genes=mutation_percent_genes,
              gene_type=float)

ga_instance.run()
ga_instance.plot_fitness(title="Fitness Evolution for Metal Alloy").savefig("Metal_fitness_evolution.png")
solution, solution_fitness, solution_idx = ga_instance.best_solution()
print("-"*50)
print("\nResults:")
print(f"Best metal composition (x, y, z, u, v, w): {solution}")
print(f"Maximum endurance: {solution_fitness}")

# Przy ponownym uruchomieniu algorytmu, model zwraca nieznacznie inne wyniki, są one jednak
# na tyle bliskie, że można uznać je za niewidoczne. Wytrzymałość materiału wacha się w granicach
# 2.81-2.85, co jest zgodne z oczekiwaniami. Same kompozycje metalu również są
# bardzo podobne, różnią się jednak troche bardziej. Zawsze blisko jedynki znajduję się Z i U.
# Przykładowy wynik - [0.59014843 0.56939783 0.99891439 0.02192851 0.94423879 0.11717282]