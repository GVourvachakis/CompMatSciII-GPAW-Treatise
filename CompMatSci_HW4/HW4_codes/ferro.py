#!/usr/bin/env python3
"""
Perform a spin-polarized DFT calculation for FM/Anti-FM/non-M bcc Fe using GPAW.

Example usage:
# For ferromagnetic with U=1.0 eV (default U value)
python ferro.py ferro

# For ferromagnetic with U=1.5 eV
python ferro.py ferro --U 1.5
"""
import os
import argparse
from ase import Atoms
from gpaw import GPAW, PW, setup_paths

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

# Global variable to define ordering
# ordering: str = "ferro" # options: ferro, antiferro, non 

def spinpol_calculation(ordering, u_value):
    # Define lattice constant in Angstroms and initial magnetic moment in Bohr magnetons.
    a = 2.87
    m = 2.2

    # Set up the magmom parameters
    if ordering == "ferro":
        magmoms: list[float,float] = [m,m]
        title: str = "Ferromagnetic"
    elif ordering == "antiferro":
        magmoms: list[float,float] = [m,-m]
        title: str = "Anti-ferromagnetic"
    elif ordering == "non":
        magmoms: list[float,float] = [0,0]
        title: str = "Non-magnetic"
    else:
        raise ValueError(f"Unsupported state: {ordering}")

    print(f"Using magnetic moments: {magmoms}")
    print(f"Using Hubbard U = {u_value} eV")

    # Create a two-atom structure for Fe.
    # The two atoms are placed at fractional coordinates (0,0,0) and (0.5, 0.5, 0.5)
    # within the cubic cell defined by (a, a, a). Periodic boundary conditions (pbc) are used.
    fe = Atoms("Fe2",
               scaled_positions=[(0, 0, 0),
                                 (0.5, 0.5, 0.5)],
               magmoms=magmoms,
               cell=(a, a, a),
               pbc=True)

    # Define the Hubbard U parameters for Fe d-orbitals
    setups = {'Fe': f':d,{u_value}'}  # Format is 'element': ':orbital,U_value'
    
    # Initialize the GPAW calculator.
    # The 'PW(500)' means a plane-wave basis set with an energy cutoff of 500 eV.
    # A 10x10x10 Monkhorst–Pack k-point grid is used.
    # Initialize the GPAW calculator with +U correction
    calc = GPAW(mode=PW(500),
               kpts=(10, 10, 10),
               setups=setups,
               txt=f"{ordering}_U{u_value}.txt")

    # Attach the calculator to the atoms.
    fe.calc = calc

    # Compute the potential energy; this triggers the SCF calculation.
    energy = fe.get_potential_energy()
    print(f"{title} calculation with U={u_value} eV converged energy: {energy:.4f} eV")
    
    # Save the final calculator state to a GPW file for later use.
    calc.write(f"{ordering}_U{u_value}.gpw")

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Perform spin-polarized DFT+U calculation for Fe")
    parser.add_argument("ordering", type=str, choices=["ferro", "antiferro", "non"],
                       help="Magnetic ordering: ferro, antiferro, or non")
    parser.add_argument("--U", type=float, default=1.0,
                       help="Hubbard U parameter in eV (default: 1.0)")
    
    # Parse command-line arguments
    args = parser.parse_args()

    # Run the calculation with the provided parameters
    spinpol_calculation(args.ordering, args.U)