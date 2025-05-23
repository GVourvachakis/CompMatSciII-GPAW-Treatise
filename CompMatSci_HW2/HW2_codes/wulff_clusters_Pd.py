# work from : https://sharc.materialsmodeling.org/wulff_construction/

#from wulffpack import SingleCrystal

import numpy as np
import matplotlib.pyplot as plt
from ase.visualize import view
from ase.io import read
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

num_atoms: int = 489
num_atoms_approx: int = 500

# Read the XYZ file
atoms = read(f'./wulff_constructions_Pd/Pd-{num_atoms}.xyz')

# Get positions and atomic numbers
positions = atoms.get_positions()
atomic_numbers = atoms.get_atomic_numbers()

# Set up the 3D plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Define a color map based on atomic numbers
# Palladium is atomic number 46
colors = ['silver' if num == 46 else 'blue' for num in atomic_numbers]

# Plot atoms as spheres
# Adjust the size based on atomic number
sizes = [100 if num == 46 else 50 for num in atomic_numbers]

# Plot the atoms
for i, (x, y, z) in enumerate(positions):
    ax.scatter(x, y, z, color=colors[i], s=sizes[i], edgecolors='black', alpha=0.7)

# Draw bonds between atoms that are close enough
# For Pd-Pd bonds, typical bond length is around 2.75 Angstroms
bond_threshold = 3.0  # Angstroms

for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        if dist < bond_threshold:
            ax.plot([positions[i][0], positions[j][0]],
                    [positions[i][1], positions[j][1]],
                    [positions[i][2], positions[j][2]],
                    color='gray', alpha=0.5)

# Set axis labels
ax.set_xlabel('X (Å)')
ax.set_ylabel('Y (Å)')
ax.set_zlabel('Z (Å)')

# Set title
ax.set_title(f'Visualization of Pd-{num_atoms_approx} Cluster')

# Make the plot more visually appealing
ax.grid(False)
ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio

# Show the plot
plt.tight_layout()
plt.show()