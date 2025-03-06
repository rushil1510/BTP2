import pygad
from PorousMediaBurner import simulate_burner, load_nox_parameters
import numpy as np
import matplotlib.pyplot as plt

# Load the optimal alpha from the file or use a default value
try:
    with open('optimization_results.txt', 'r') as f:
        for line in f:
            if line.startswith('Optimal alpha:'):
                alpha = float(line.split(':')[1].strip())
                break
except:
    # Default alpha if file not found
    alpha = 10
    print(f"Using default alpha value: {alpha}")
else:
    print(f"Using optimized alpha value: {alpha}")

# Define constraints
MIN_TEMPERATURE = 300  # Minimum acceptable temperature (K)
MAX_NOX = 100          # Maximum acceptable NOx (ppm)

# Updated fitness function with constraints
def fitness_function(ga_instance, solution, solution_idx):
    porosity_SiC3, porosity_SiC10, preheating_length = solution
    temperature, NOx = simulate_burner(porosity_SiC3, porosity_SiC10, preheating_length)
    
    # Check if solution violates constraints
    if temperature < MIN_TEMPERATURE or NOx > MAX_NOX or temperature == -1e6 or NOx == 1e6:
        # Return a very negative fitness for invalid solutions
        print(f"Solution {solution_idx}: {solution} REJECTED - Constraints violated (T={temperature:.1f}K, NOx={NOx:.2f})")
        return -10000  # Strong penalty for invalid solutions
    
    # Define fitness as maximizing temperature while minimizing NOx for valid solutions
    fitness = temperature - alpha * NOx
    print(f"Solution {solution_idx}: {solution}, Temp: {temperature:.1f}K, NOx: {NOx:.2f}, Fitness: {fitness:.2f}")
    return fitness

# Define the parameter space
gene_space = [
    {'low': 0.75, 'high': 0.85},  # SiC3 porosity
    {'low': 0.75, 'high': 0.85},  # SiC10 porosity
    {'low': 0.02, 'high': 0.04}   # Preheating length (meters)
]

# Custom callback function to monitor and guide optimization
def on_generation(ga_instance):
    print(f"Generation = {ga_instance.generations_completed}")
    print(f"Best solution fitness = {ga_instance.best_solution()[1]}")
    
    # Check if best solution meets constraints
    best_solution = ga_instance.best_solution()[0]
    temperature, NOx = simulate_burner(best_solution[0], best_solution[1], best_solution[2])
    
    if temperature < MIN_TEMPERATURE or NOx > MAX_NOX or temperature == -1e6 or NOx == 1e6:
        print("Warning: Best solution still violates constraints")
    
    return False  # Return False to continue evolution

# Create and configure the genetic algorithm
ga_instance = pygad.GA(
    num_generations=20, 
    num_parents_mating=5,
    fitness_func=fitness_function, 
    sol_per_pop=10,
    num_genes=3, 
    gene_space=gene_space,
    mutation_num_genes=1,
    mutation_probability=0.02,
    on_generation=on_generation,
    save_best_solutions=True
)

# Run the optimization
print(f"Starting optimization with alpha = {alpha}")
print(f"Constraints: Min Temperature = {MIN_TEMPERATURE}K, Max NOx = {MAX_NOX} units")
ga_instance.run()

# Get the best solution
solution, solution_fitness, solution_idx = ga_instance.best_solution()
porosity_SiC3, porosity_SiC10, preheating_length = solution
temperature, NOx = simulate_burner(porosity_SiC3, porosity_SiC10, preheating_length)

# Verify the final solution meets constraints
constraints_met = (temperature >= MIN_TEMPERATURE and NOx <= MAX_NOX and 
                  temperature != -1e6 and NOx != 1e6)

# Print results
print("\nOPTIMIZATION RESULTS:")
print(f"Optimized Parameters: Porosity SiC3 = {porosity_SiC3:.4f}, Porosity SiC10 = {porosity_SiC10:.4f}, Preheating Length = {preheating_length:.4f} m")
print(f"Optimized Fitness: {solution_fitness}")
print(f"Temperature: {temperature:.2f} K")
print(f"NOx Emissions: {NOx:.2f} units")
print(f"Alpha value used: {alpha}")
print(f"Constraints met: {constraints_met}")

# Plot fitness progression
plt.figure(figsize=(10, 6))
plt.plot(ga_instance.best_solutions_fitness, 'b-', label='Best Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Optimization Progress')
plt.grid(True)
plt.legend()
plt.savefig('optimization_progress.png')

# Save detailed results to a file
with open('final_results.txt', 'w') as f:
    f.write(f"Optimal alpha: {alpha}\n")
    f.write(f"Optimized Parameters:\n")
    f.write(f"  - Porosity SiC3: {porosity_SiC3:.6f}\n")
    f.write(f"  - Porosity SiC10: {porosity_SiC10:.6f}\n")
    f.write(f"  - Preheating Length: {preheating_length:.6f} m\n")
    f.write(f"Optimized Results:\n")
    f.write(f"  - Fitness: {solution_fitness:.6f}\n")
    f.write(f"  - Temperature: {temperature:.2f} K\n")
    f.write(f"  - NOx Emissions: {NOx:.2f} units\n")
    f.write(f"  - Constraints Met: {constraints_met}\n")
    
    # Load NOx model parameters
    c1, c2 = load_nox_parameters()
    f.write(f"\nNOx Model Parameters:\n")
    f.write(f"  - c1: {c1:.6f}\n")
    f.write(f"  - c2: {c2:.1f}\n")