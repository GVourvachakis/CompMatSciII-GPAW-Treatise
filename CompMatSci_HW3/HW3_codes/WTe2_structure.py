import numpy as np
from ase import Atoms
from pymatgen.core.structure import Structure

# Step 1: Building the 2H-WTe₂ structure
def build_wte2_structure(a=3.496, c=14.07, vacuum=0.0):
    """
    Create the 2H-WTe₂ structure with initial lattice parameters.
    The structure has hexagonal symmetry with space group P6₃/mmc.

    Args:
        a (float): a lattice parameter in Å
        c (float): c lattice parameter in Å
        vacuum (float): Vacuum padding in z-direction (set to 0.0 since we study 3D properties)

    Returns:
        ASE Atoms object of 2H-WTe₂
    """
    # 2H-WTe₂ structure parameters
    # Hexagonal lattice: P6₃/mmc symmetry — 2 W + 4 Te in unit cell

    # Construct hexagonal lattice vectors ({(a2, a3), (a3, a1), (a1, a2)} -> {(100), (010), (001)})
    cell = [
                [a / 2, -a * np.sqrt(3) / 2, 0], # a1
                [a / 2,  a * np.sqrt(3) / 2, 0], # a2
                [0, 0, c + vacuum]               # a3
            ]

    # Accurate fractional z-position for Te from Materials Project (mp-2815)
    z_Te = 0.62

    # Scaled fractional positions
    positions = [
        (1/3, 2/3, 1/4),          # W
        (2/3, 1/3, 3/4),          # W
        (1/3, 2/3, 0.25 + z_Te),  # Te
        (2/3, 1/3, 0.25 - z_Te),  # Te
        (1/3, 2/3, 0.75 - z_Te),  # Te
        (2/3, 1/3, 0.75 + z_Te)   # Te
    ]

    # Create the structure
    symbols = ['W', 'W', 'Te', 'Te', 'Te', 'Te']
    wte2 = Atoms(symbols=symbols,
                 scaled_positions=positions,
                 cell=cell,
                 pbc=[True, True, True] # Fully periodic for bulk
                 )
    
    if vacuum > 0:
        wte2.center(axis=2)  # Center slab in vacuum along z - for surface analysis
    
    return wte2

# Alternatively, load from Materials Project using pymatgen (not used by default)
def get_structure_from_mp():
    """
    Alternative method to get structure from Materials Project data.
    This is useful if you have the MP ID and want to use their optimized structure.

    Returns:
        ASE Atoms object converted from pymatgen Structure
    """
    # For actual implementation, you would use the MP API
    # Here we'll create it using pymatgen Structure directly based on known data
    lattice = [[1.748, -3.028, 0],
               [1.748, 3.028, 0],
               [0, 0, 14.07]]
    species = ["W", "W", "Te", "Te", "Te", "Te"]
    coords = [
        [1/3, 2/3, 1/4],
        [2/3, 1/3, 3/4],
        [1/3, 2/3, 0.12],
        [2/3, 1/3, 0.38],
        [1/3, 2/3, 0.62],
        [2/3, 1/3, 0.88]
    ]

    structure = Structure(lattice, species, coords)

    # Convert to ASE Atoms
    from pymatgen.io.ase import AseAtomsAdaptor
    atoms = AseAtomsAdaptor.get_atoms(structure)

    return atoms


if __name__ == "__main__":
    from ase.visualize import view
    atoms = build_wte2_structure()
    view(atoms)