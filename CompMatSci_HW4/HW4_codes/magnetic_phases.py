#!/usr/bin/env python3
"""
Compare the energies of three magnetic phases (ferromagnetic, antiferromagnetic, and nonmagnetic)
for bcc Fe using pre-calculated GPW files with Hubbard U correction. The ferromagnetic state is 
taken as the reference (0 eV), and energy differences are computed for both LDA+U and an estimate 
of PBE via an xc difference correction. Additionally, the script prints out the calculated magnetic 
moments for the ferromagnetic configuration.

Prerequisites:
 - The GPW files '{ordering}_U{u_value}.gpw' must exist in the working directory.
 - These files are obtained from running the ferro.py script with different ordering modes
   and U values (e.g., 'python ferro.py ferro --U 1.0').
 - Supported ordering values: 'ferro', 'antiferro', 'non'

Usage:
 - Run 'python magnetic_phases.py --U 1.0' to compare results with U=1.0 eV
 - Default U value is 1.0 eV if not specified
"""
import argparse
from gpaw import GPAW
from ferro import setup_gpaw_paths

setup_gpaw_paths()

def magnetic_phases(u_value):
    print(f"state LDA+U(U={u_value}) PBE")

    # Initialize reference energies with the ferromagnetic state.
    eLDA0 = None
    ePBE0 = None

    # Define the list of magnetic state names.
    # These correspond to files 'ferro.gpw', 'anti.gpw', and 'non.gpw'.
    states = ['ferro', 'antiferro', 'non']

    for state in states:
        # Load the saved calculator from the .gpw file for the current state.
        calc = GPAW(f"{state}_U{u_value}.gpw", txt=None)
        # Extract the Atoms object from the calculator.
        atoms = calc.get_atoms()
        # Get the potential energy computed using LDA (the original calculation).
        eLDA = atoms.get_potential_energy()
        # Compute the difference in the exchange-correlation energy between PBE and LDA.
        deltaxc = calc.get_xc_difference('PBE')
        ePBE = eLDA + deltaxc
        # For the ferromagnetic state, save the reference energy.
        if state == 'ferro':
            eLDA0 = eLDA
            ePBE0 = ePBE
        # Compute energy differences relative to the ferromagnetic state.
        eLDA_diff = eLDA - eLDA0
        ePBE_diff = ePBE - ePBE0
        # Print the results formatted with state name and energy differences.
        print(f"{state:<5s}: {eLDA_diff:7.3f} eV {ePBE_diff:7.3f} eV")

def fm_moments(u_value):
    """
    Load the ferromagnetic calculation and print the magnetic moments for each atom.
    """
    # Load the ferromagnetic calculator state.
    calc = GPAW(f"ferro_U{u_value}.gpw", txt=None)
    # Retrieve the Atoms object.
    atoms = calc.get_atoms()
    # Get the magnetic moments from the spin density.
    magmoms = atoms.get_magnetic_moments()
    print(f"Calculated magnetic moments for ferromagnetic Fe (per atom) with U={u_value} eV:", magmoms)

if __name__ == "__main__":
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(description="Compare magnetic phases with DFT+U")
    parser.add_argument("--U", type=float, default=1.0,
                      help="Hubbard U parameter used in calculations (default: 1.0)")
    
    args = parser.parse_args()
    
    magnetic_phases(args.U)
    fm_moments(args.U)