import pygad
from PorousMediaBurner import simulate_burner

# Updated fitness function
def fitness_function(ga_instance, solution, solution_idx):
    porosity_SiC3, porosity_SiC10, preheating_length = solution
    temperature, NOx = simulate_burner(porosity_SiC3, porosity_SiC10, preheating_length)
    
    # Define fitness as maximizing temperature while minimizing NOx
    return temperature - 100 * NOx  # Adjust the weight for NOx penalty if needed
gene_space = [
    {'low': 0.75, 'high': 0.85},  # SiC3 porosity
    {'low': 0.75, 'high': 0.85},  # SiC10 porosity
    {'low': 0.02, 'high': 0.04}   # Preheating length (meters)
]

ga_instance = pygad.GA(
    num_generations=20, num_parents_mating=5,
    fitness_func=fitness_function, sol_per_pop=10,
    num_genes=3, gene_space=gene_space,
    mutation_num_genes=1, mutation_probability=0.02  # Reduce mutation rate
)
ga_instance.run()

solution, solution_fitness, _ = ga_instance.best_solution()
print(f"Optimized Parameters: {solution}")
print(f"Optimized Fitness: {solution_fitness}")