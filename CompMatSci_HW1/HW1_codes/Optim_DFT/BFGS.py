from gpaw import GPAW, PW, setup_paths

from ase.build.bulk import bulk
from ase.visualize import view
from ase.optimize import BFGS
from ase.io import write, read
from ase.io.trajectory import Trajectory

import matplotlib.pyplot as plt
import numpy as np
import os
import sys


# Suppress detailed GPAW output
os.environ['GPAW_VERBOSE'] = '0'
sys.stdout = open(os.devnull, 'w') # Redirect standard output

# Clear existing paths and set the new one [user-dependent]
intended_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
setup_paths[:] = [intended_path] # Replace all existing paths
os.environ['GPAW_SETUP_PATH'] = intended_path
sys.stdout = sys.stdout # Restore standard output

print("GPAW looking for datasets in:", setup_paths)
print("Environment GPAW_SETUP_PATH:", os.environ['GPAW_SETUP_PATH'])

# Create the aluminum fcc structure
atoms = bulk("Al", cubic=True)
original_positions = atoms.positions.copy()
print("\nOriginal atomic positions:")
print(original_positions)

# Displace one of the atoms
my_y = 0.6
atoms[1].y = my_y
displaced_positions = atoms.positions.copy()
print(f"\n---------DATATYPE OF POSITIONS:\t{type(displaced_positions)}------------\n")

print("\nDisplaced atomic positions (before optimization):")
print(displaced_positions)
print(f"Atom 1 y-coordinate modified from {original_positions[1][1]:.4f} to {my_y:.4f}")

# Set up the calculator
atoms.calc = GPAW(mode=PW(300), kpts=(4, 4, 4), txt="gpaw.log")

# Calculate initial energy
initial_energy = atoms.get_potential_energy()
initial_forces = atoms.get_forces()

print(f"\nInitial energy: {initial_energy:.6f} eV")
print("Initial forces (eV/Å):")
for i, force in enumerate(initial_forces):
    print(f"Atom {i}: {force}")

# Largest force magnitude
max_force_initial = np.max(np.sqrt(np.sum(initial_forces**2, axis=1)))
print(f"Maximum force: {max_force_initial:.6f} eV/Å")

# Save the initial structure
write('al_initial.xyz', atoms)

# Create a figure to track optimization
plt.figure(figsize=(12, 10))

# Run BFGS optimization
print("\nStarting BFGS optimization...")

# Initialize BFGS optimizer with trajectory file
dyn = BFGS(atoms, trajectory='al.traj')

# Capture optimization data
energies = []
max_forces = []
displacements = []
step_lengths = []

# Define a observer function
def get_optimization_data():
    energies.append(atoms.get_potential_energy())
    forces = atoms.get_forces()
    max_forces.append(np.max(np.sqrt(np.sum(forces**2, axis=1))))
    
    # Calculate displacement from original
    current_pos = atoms.positions.copy()
    disp = np.linalg.norm(current_pos - displaced_positions)
    displacements.append(disp)
    
    # Calculate step length (if not the first step)
    if len(displacements) > 1:
        step = np.linalg.norm(current_pos - previous_pos)
        step_lengths.append(step)
    
    return current_pos

# Add first data point
previous_pos = atoms.positions.copy()
get_optimization_data()

# Attach observer and run optimization
dyn.attach(lambda: get_optimization_data(), interval=1)
dyn.run(fmax=0.05)  # Run until maximum force is less than 0.05 eV/Å

# Calculate final data
final_energy = atoms.get_potential_energy()
final_forces = atoms.get_forces()
final_positions = atoms.positions.copy()

# Calculate step lengths for the plot
if len(step_lengths) < len(energies) - 1:  # Add last step if missing
    step_lengths.append(np.linalg.norm(final_positions - previous_pos))

# Print optimization results
print("\nBFGS optimization completed!")
print(f"Number of steps: {len(energies)-1}")
print(f"Final energy: {final_energy:.6f} eV")
print(f"Energy change: {final_energy - initial_energy:.6f} eV")
print("\nFinal atomic positions:")
print(final_positions)
print("\nFinal forces (eV/Å):")
for i, force in enumerate(final_forces):
    print(f"Atom {i}: {force}")

# Calculate displacement from original positions
total_displacement = np.linalg.norm(final_positions - displaced_positions)
print(f"\nTotal atomic displacement during optimization: {total_displacement:.6f} Å")

# Calculate displacement of the previously modified atom
atom1_displacement = np.linalg.norm(final_positions[1] - displaced_positions[1])
print(f"Atom 1 displacement: {atom1_displacement:.6f} Å")
print(f"Atom 1 y-coordinate changed from {my_y:.4f} to {final_positions[1][1]:.4f}")

# Save final structure
write('al_final.xyz', atoms)

# Plotting the optimization data
iterations = range(len(energies))

# Plot energy vs. iteration
plt.subplot(2, 2, 1)
plt.plot(iterations, energies, 'o-', markersize=4, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Energy (eV)')
plt.title('Energy vs. Iteration')
plt.grid(True)

# Plot maximum force vs. iteration
plt.subplot(2, 2, 2)
plt.plot(iterations, max_forces, 'o-', markersize=4, linewidth=2, color='red')
plt.axhline(y=0.05, color='k', linestyle='--', label='Convergence threshold')
plt.xlabel('Iteration')
plt.ylabel('Maximum Force (eV/Å)')
plt.title('Maximum Force vs. Iteration')
plt.grid(True)
plt.legend()

# Plot cumulative displacement vs. iteration
plt.subplot(2, 2, 3)
plt.plot(iterations, displacements, 'o-', markersize=4, linewidth=2, color='green')
plt.xlabel('Iteration')
plt.ylabel('Displacement from Initial (Å)')
plt.title('Cumulative Displacement vs. Iteration')
plt.grid(True)

# Plot step length vs. iteration
plt.subplot(2, 2, 4)
step_iter = range(len(step_lengths))
plt.plot(step_iter, step_lengths, 'o-', markersize=4, linewidth=2, color='purple')
plt.xlabel('Iteration')
plt.ylabel('Step Length (Å)')
plt.title('Step Length vs. Iteration')
plt.grid(True)

plt.tight_layout()
plt.savefig('bfgs_optimization.png', dpi=300)
print("\nOptimization plots saved as 'bfgs_optimization.png'")

# Create visualization of the optimization trajectory
print("\nCreating trajectory visualization...")
traj = Trajectory('al.traj')

# Create a plot showing the path of the displaced atom
plt.figure(figsize=(10, 8))
positions = np.array([atoms[1].position for atoms in traj])

# 3D plot of atom trajectory
from mpl_toolkits.mplot3d import Axes3D
ax = plt.figure(figsize=(10, 8)).add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o-', markersize=4, linewidth=2)
ax.plot([positions[0, 0]], [positions[0, 1]], [positions[0, 2]], 'ro', markersize=8, label='Start')
ax.plot([positions[-1, 0]], [positions[-1, 1]], [positions[-1, 2]], 'go', markersize=8, label='End')

# Mark the y-coordinate that was manually changed
ax.plot([positions[0, 0]], [my_y], [positions[0, 2]], 'bo', markersize=8, label='Manually set y')

ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')
ax.set_title('Trajectory of Displaced Atom During BFGS Optimization')
ax.legend()
plt.savefig('atom_trajectory.png', dpi=300)
print("Atom trajectory saved as 'atom_trajectory.png'")

# View the initial and final structures
view(read('al_initial.xyz'))
view(read('al_final.xyz'))







# from ase.optimize import BFGS
# dyn = BFGS(atoms, trajectory='al.traj')
# dyn.run(fmax=0.05)
# print(atoms.positions)
# view(atoms)