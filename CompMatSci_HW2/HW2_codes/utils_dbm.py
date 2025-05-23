# Visualization routines for coordination numbers: dangling_bonds_plot_2D(), coordination_distribution()

import numpy as np
from ase.neighborlist import NeighborList
import matplotlib.pyplot as plt

material: str = "Pd"

def coordination_and_area(slab):
    # Calculate surface area (a1 x a2)
    a1 = slab.cell[0]
    a2 = slab.cell[1]
    area = a1[0]*a2[1] - a1[1]*a2[0]  # in Å²
    
    # Calculate coordination numbers using ASE's NeighborList
    # For Al, cutoff distance is about 3.0 Å [see ase.neighborlist.natural_cutoffs]
    cutoff = 3.0
    nl = NeighborList([cutoff/2] * len(slab), self_interaction=False, bothways=True)
    nl.update(slab)
    
    coordination_numbers = []
    for i in range(len(slab)):
        indices, offsets = nl.get_neighbors(i)
        coordination_numbers.append(len(indices))
    
    return coordination_numbers, area

def dangling_bonds_plot_2D(slab, title: str = material, surface_type: str | None = None, z_b=12):
    """
    Create a simple 2D top-down plot of the slab, highlighting surface atoms
    and coloring them by coordination number.
    """
    positions = slab.positions
    coord_numbers, _ = coordination_and_area(slab)  # function assumed defined

    # Identify surface atoms
    if surface_type != "211":
        top_layer_z = np.max(positions[:, 2]) - 0.1
        surface_atoms = [i for i, pos in enumerate(positions) if pos[2] > top_layer_z]
    else:
        sorted_indices = np.argsort(positions[:, 2])
        z_values = positions[sorted_indices, 2]
        unique_z = []
        for z in z_values:
            if not unique_z or abs(z - unique_z[-1]) > 0.5:
                unique_z.append(z)
        top_layers = unique_z[-3:]
        surface_atoms = [i for i, pos in enumerate(positions) 
                         if any(abs(pos[2] - z) < 0.5 for z in top_layers)]
    
    surface_indices = set(surface_atoms)
    # Bulk vs. surface
    bulk_indices = [i for i in range(len(positions)) if i not in surface_indices]

    fig, ax = plt.subplots(figsize=(6, 5))

    # Plot bulk atoms in a simple color (gray)
    bulk_positions = positions[bulk_indices]
    ax.scatter(bulk_positions[:, 0], bulk_positions[:, 1],
               color='gray', alpha=0.5, label='Bulk Atoms')

    # Plot surface atoms, color-coded by coordination
    surface_positions = positions[surface_atoms]
    surface_coords = [coord_numbers[i] for i in surface_atoms]

    # Create colormap
    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(surface_coords), vmax=z_b)

    sc = ax.scatter(surface_positions[:, 0], surface_positions[:, 1],
                    c=surface_coords, cmap=cmap, norm=norm,
                    edgecolor='k', s=80, label='Surface Atoms')

    # Optional: annotate surface atoms with # of dangling bonds
    # (z_b - coordination)
    for i in surface_atoms:
        x, y, _ = positions[i]
        dbonds = z_b - coord_numbers[i]
        ax.text(x, y, f"{dbonds}", fontsize=7,
                horizontalalignment='center', verticalalignment='center')

    ax.set_title(f"{title}({surface_type}) - 2D Top-down View")
    ax.set_xlabel("X (Å)")
    ax.set_ylabel("Y (Å)")

    # Colorbar for coordination
    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label("Coordination Number")

    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def coordination_distribution(slab, title: str = material, surface_type: str | None = None, z_b=12):
    """
    Plot a histogram of the coordination numbers (or dangling bonds) for surface atoms.
    """
    positions = slab.positions
    coord_numbers, _ = coordination_and_area(slab)

    # Identify surface atoms (same logic as above)
    if surface_type != "211":
        top_layer_z = np.max(positions[:, 2]) - 0.1
        surface_atoms = [i for i, pos in enumerate(positions) if pos[2] > top_layer_z]
    else:
        sorted_indices = np.argsort(positions[:, 2])
        z_values = positions[sorted_indices, 2]
        unique_z = []
        for z in z_values:
            if not unique_z or abs(z - unique_z[-1]) > 0.5:
                unique_z.append(z)
        top_layers = unique_z[-3:]
        surface_atoms = [i for i, pos in enumerate(positions) 
                         if any(abs(pos[2] - z) < 0.5 for z in top_layers)]
    
    # Get coordination and dangling bonds for surface atoms
    surface_coords = [coord_numbers[i] for i in surface_atoms]
    dangling_bonds = [z_b - c for c in surface_coords]

    # Plot histogram of coordination numbers
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(surface_coords, bins=range(0, z_b+2), color='blue', alpha=0.7, edgecolor='black')
    ax.set_title(f"{title}({surface_type}) - Surface Coordination Distribution")
    ax.set_xlabel("Coordination Number")
    ax.set_ylabel("Count of Surface Atoms")
    plt.tight_layout()
    plt.show()

    # Optionally, also plot dangling bonds distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(dangling_bonds, bins=range(0, (z_b+1)), color='red', alpha=0.7, edgecolor='black')
    ax.set_title(f"{title}({surface_type}) - Surface Dangling Bonds Distribution")
    ax.set_xlabel("Dangling Bonds (z_b - coordination)")
    ax.set_ylabel("Count of Surface Atoms")
    plt.tight_layout()
    plt.show()