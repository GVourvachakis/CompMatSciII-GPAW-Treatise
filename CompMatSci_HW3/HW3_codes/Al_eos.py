from __future__ import print_function
from ase.eos import EquationOfState
from gpaw import GPAW, PW
import numpy as np
import matplotlib.pyplot as plt
# ASE's built-in crystal generators for accurate structures
from ase.lattice.cubic import FaceCenteredCubic, BodyCenteredCubic
from Al_structure import create_bulk_structure, get_lattice_constant, setup_gpaw_paths

# Global variable to define the crystal structure
structure = "bcc"  # Options: "fcc" or "bcc"

# optional (not used here): for comparison purposes, though it changes the calculations. 
# Using ASE's built-in crystal structure generator.
def conventional_bulk_structure(alpha):
    """Create conventional unit cell for FCC/BCC Aluminium."""
    # Structure   Al_structure setup  Established Implementation Atoms/Cell Volumes/Atom
    # FCC	     Primitive (1 atom)	  Conventional (4 atoms)	    4	        a³/4
    # BCC	     Non-standard cell	  Conventional (2 atoms)	    2	        a³/2
    
    if structure == "fcc":
        # Conventional FCC cell (4 atoms)
        return FaceCenteredCubic('Al', latticeconstant=alpha, size=(1,1,1))
    elif structure == "bcc":
        # Conventional BCC cell (2 atoms)
        return BodyCenteredCubic('Al', latticeconstant=alpha, size=(1,1,1))
    else:
        raise ValueError(f"Unsupported structure: {structure}")

def compute_energies_vs_volume(alpha0: float, num_points: int = 20):
    """Compute potential energies for different volumes around the equilibrium lattice constant using primitive cells."""
    # Create the primitive cell at the experimental lattice constant.
    bulk = create_bulk_structure(alpha0)

    # Since the primitive cell contains one atom, the volume per atom is just the cell volume.
    V0 = bulk.get_volume() 

    # variations based on lattice constants. Must be transformed to volume variations.
    if structure == "fcc": 
        variation = 0.10  # 10% relative variation for FCC (better adaptive fit)
    elif structure == "bcc": 
        variation = 0.15  # Adjust if a narrower range is preferred for BCC
        # Handling of warning: 
        # UserWarning: The minimum volume of your fit is not in your volumes.  
        # You may not have a minimum in your dataset!
    else: raise ValueError("Unsupported crystal structure")

    # Generate the range of volumes (per atom) using cubic scaling
    volume_min = V0 * (1 - variation)**3
    volume_max = V0 * (1 + variation)**3
    volume_points = np.linspace(volume_min, volume_max, num_points)
    
    print(f"Range: V0*{(1 - variation)**3}, V0*{ (1 + variation)**3}\n")
    print(f"Volume range: {volume_min:.3f} Å³ to {volume_max:.3f} Å³ (per primitive cell)\n")
    
    # Generate volume range
    volumes = []
    energies = []
    
    # results from Elbow method (E vs. k curve):
    if structure=="fcc": k_elbow=4
    elif structure=="bcc": k_elbow=7
    
    for vol in volume_points:
        # Determine the new lattice constant corresponding to the target volume per atom.
        # For a conventional cell, the relation is:
        #   For FCC: V_atom = a^3/4  -> a = (4 * V_atom)^(1/3)
        #   For BCC: V_atom = a^3/2  -> a = (2 * V_atom)^(1/3)
        # This can be written in a unified way using V0 = a0^3/n (with n atoms per cell)
        # Then: a = a0 * ( (vol / V0) )^(1/3)
        # Scale lattice constant cubically

        # Update the lattice constant corresponding to the target primitive cell volume.
        alpha = alpha0 * (vol / V0)**(1/3)

        # Create the structure with the updated lattice constant
        bulk = create_bulk_structure(alpha)
        
        # Set up the GPAW calculator with improved parameters
        calc = GPAW(mode=PW(300),       
                    kpts=(k_elbow,k_elbow,k_elbow),
                    txt=None,            
                    occupations={'name': 'fermi-dirac', 'width': 0.1},
                    basis={'Al': 'dzp'})  # More accurate basis set
        
        bulk.calc = calc # Assign the calculator to the Atoms object
        
        # Compute energy and volume (for the primitive cell)
        energy = bulk.get_potential_energy()
        cell_volume = bulk.get_volume()
        
        volumes.append(cell_volume)
        energies.append(energy)
        
        print(f"Volume: {cell_volume:.3f} Å³, Lattice Constant: {alpha:.3f} Å, Energy: {energy:.6f} eV")

    return volumes, energies

def main():
    """Main routine for polymorphs of bulk Aluminium test."""
    setup_gpaw_paths()
    alpha0 = get_lattice_constant()

    num_points = 10 # Number of volume points

    # Compute the energy-volume curve using the volume-based approach
    volumes, energies = compute_energies_vs_volume(alpha0, num_points)
    
    # Use ASE Equation of State to fit and analyze the energy-volume data
    eos = EquationOfState(volumes, energies, eos='murnaghan')
    
    # Fit and extract parameters
    v0, e0, B = eos.fit()

    # Convert the bulk modulus to GPa (conversion factor: 1 eV/Å³ ≈ 160.21766309 GPa)
    B_GPa = B * (160.21766309)

    # Results output
    print(f"\nFinal Results for {structure.upper()} Aluminium:")
    print(f"Equilibrium Volume (v0): {v0:.3f} Å³ (per primitive cell)")
    print(f"Equilibrium Energy (e0): {e0:.6f} eV")
    print(f"Calculated Bulk Modulus (B (GPa)): {B_GPa:.2f} GPa")
 
    # Plot the Equation of State
    plt.figure(figsize=(10, 6))
    eos.plot(filename=f'eos_{structure}.png')
    
    # Compare with experimental values
    if structure == "fcc":
        exp_B = 76.2  # GPa
    elif structure == "bcc":
        exp_B = 475.0  # GPa
    else:
        raise ValueError("Unsupported structure")
    
    perc_error = 100 * abs(B_GPa - exp_B) / exp_B
    print(f"Experimental Bulk Modulus: {exp_B} GPa")
    print(f"Percentage Error: {perc_error:.5f}%")

if __name__ == '__main__':
    main()