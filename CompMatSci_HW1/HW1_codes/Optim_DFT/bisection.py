from gpaw import GPAW, PW, setup_paths
import ase
from ase.build.bulk import bulk
from ase.visualize import view

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

# Suppress detailed GPAW output
os.environ['GPAW_VERBOSE'] = '0'
sys.stdout = open(os.devnull, 'w')  # Redirect standard output

# Clear existing paths and set the new one
intended_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
setup_paths[:] = [intended_path]  # Replace all existing paths
os.environ['GPAW_SETUP_PATH'] = intended_path

sys.stdout = sys.__stdout__  # Restore standard output
print("GPAW looking for datasets in:", setup_paths)
print("Environment GPAW_SETUP_PATH:", os.environ['GPAW_SETUP_PATH'])

# Create the fcc Al structure
atoms = bulk("Al", cubic="True")
print("Original atomic positions:")
print(atoms.positions)

# Function to calculate energy for a given y-coordinate
def calculate_energy(y_coord: float) -> Tuple[np.float64, ase.atoms.Atoms]:
    atoms_copy = atoms.copy()
    atoms_copy[1].y = y_coord
    atoms_copy.calc = GPAW(mode=PW(300), kpts=(4, 4, 4), txt="gpaw.log")
    energy = atoms_copy.get_potential_energy()
    return energy, atoms_copy

# Initial test with the given displacement
my_y = 0.6
initial_energy, displaced_atoms = calculate_energy(my_y)
print(f"Initial position (y={my_y}): Energy = {initial_energy:.6f} eV")
print("Displaced atomic positions:")
print(displaced_atoms.positions)

#  Implement bisection method to find minimum energy
# Define initial interval [a, b]
a = 0.5  # Lower bound
b = 3.0  # Upper bound
tol = 0.1  # Desired accuracy in eV
max_iterations = 20  # Safety limit

# Store results for visualization
iterations = []
y_values = []
energies = []

print("\nStarting bisection method to find minimum energy position:")
print(f"{'Iteration':^10}{'Left y':^10}{'Middle y':^10}{'Right y':^10}{'Energy(L)':^12}{'Energy(M)':^12}{'Energy(R)':^12}")
print("-" * 80)

iteration = 0
while (b - a) > tol and iteration < max_iterations:
    iteration += 1
    
    # Calculate midpoint
    c = (a + b) / 2
    
    # Calculate energies at points a, c, and b
    energy_a, _ = calculate_energy(a)
    energy_c, _ = calculate_energy(c)
    energy_b, _ = calculate_energy(b)
    
    # Store results
    iterations.append(iteration)
    y_values.append([a, c, b])
    energies.append([energy_a, energy_c, energy_b])
    
    # Print current state
    print(f"{iteration:^10d}{a:^10.4f}{c:^10.4f}{b:^10.4f}{energy_a:^12.4f}{energy_c:^12.4f}{energy_b:^12.4f}")
    
    # Update interval based on energies
    if energy_a > energy_c < energy_b:
        # If midpoint has lowest energy, narrow from both sides
        a_new = (a + c) / 2
        b_new = (c + b) / 2
        
        # Calculate new energies
        energy_a_new, _ = calculate_energy(a_new)
        energy_b_new, _ = calculate_energy(b_new)
        
        # Determine new interval based on energies
        if energy_a_new < energy_c:
            b = c
        elif energy_b_new < energy_c:
            a = c
        else:
            a = a_new
            b = b_new
    elif energy_a < energy_c:
        b = c  # Minimum is in left half
    else:
        a = c  # Minimum is in right half

# Final calculation at the converged position
final_y = (a + b) / 2
final_energy, final_atoms = calculate_energy(final_y)

print("\nBisection method converged!")
print(f"Optimal y-coordinate: {final_y:.4f}")
print(f"Minimum energy: {final_energy:.6f} eV")
print(f"Energy at original position (y={my_y}): {initial_energy:.6f} eV")
print(f"Energy reduction: {initial_energy - final_energy:.6f} eV")

# Visualization of the optimization process
plt.figure(figsize=(12, 8))

# Plot energy vs iteration
plt.subplot(2, 1, 1)
for i in range(len(iterations)):
    plt.plot([iterations[i]]*3, energies[i], 'o-', markersize=6)
plt.xlabel('Iteration')
plt.ylabel('Energy (eV)')
plt.title('Energy Evolution during Bisection Method')
plt.grid(True)

# Plot interval reduction
plt.subplot(2, 1, 2)
left_bounds = [y[0] for y in y_values]
right_bounds = [y[2] for y in y_values]
plt.fill_between(iterations, left_bounds, right_bounds, alpha=0.3, color='gray')
plt.plot(iterations, [(y[0] + y[2])/2 for y in y_values], 'r-o', label='Midpoint')
plt.xlabel('Iteration')
plt.ylabel('y-coordinate')
plt.title('Search Interval Reduction')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('bisection_optimization.png')
print("\nOptimization plot saved as 'bisection_optimization.png'")

# Visualize the final structure
view(final_atoms)