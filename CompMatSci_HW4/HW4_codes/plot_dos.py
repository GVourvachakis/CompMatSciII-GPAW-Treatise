#!/usr/bin/env python3
"""
This module reads a GPAW GPW file (in the form '{ordering}_U{u_value}.gpw'),
computes the density of states (DOS) with a specified smearing width, and plots
the results using matplotlib. Also integrates the DOS up to the Fermi level to 
estimate the net magnetic moment.

Prerequisites:
 - The GPW file '{ordering}_U{u_value}.gpw' must exist in the working directory.
 - This file is obtained from running the ferro.py script with custom ordering modes
   and U values (e.g., 'python ferro.py ferro --U 1.0').
 - Supported ordering values: 'ferro', 'antiferro', 'non'.

Usage:
    $ ./plot_dos.py <gpw_filename> [width]

Arguments:
    <gpw_filename>  Filename of the saved GPAW calculator state.
    [width]         (Optional) Smearing width in eV (default is 0.1).

The script checks the number of spin channels:
- If two spins are present, it plots DOS for 'up' and 'down' spins separately.
- Otherwise, a single DOS is plotted.

The x-axis is the energy relative to the Fermi level.

After running the script for the three cases:
    - Ferromagnetic: one expects an imbalance between the up and down DOS (non-zero net moment).
    - Antiferromagnetic: the spin-resolved DOS are nearly identical (net moment is zero).
    - Nonmagnetic: only one (degenerate) DOS is present.
"""

import sys
import matplotlib.pyplot as plt
import numpy as np
from gpaw import GPAW
from gpaw.dos import DOSCalculator

from ferro import setup_gpaw_paths

setup_gpaw_paths()

def main()->None:
    # Parse command-line arguments.
    if len(sys.argv) < 2:
        sys.exit(f"Usage: {sys.argv[0]} <gpw_filename> [width]")

    filename = sys.argv[1]
    try:
        width = float(sys.argv[2]) if len(sys.argv) > 2 else 0.1
    except ValueError:
        sys.exit("Error: width must be a number (in eV).")

    # Load the calculator from the provided GPW file.
    # Using GPAW() loads the previously stored calculator state.
    calc = GPAW(filename)
    # Ensure the electron density and eigenvalues are loaded by querying the potential energy.
    calc.get_potential_energy()

    # Load the saved calculator state.
    calc = GPAW(filename)
    # Ensure data is loaded.
    calc.get_potential_energy()

    # Initialize the DOSCalculator.
    dos = DOSCalculator.from_calculator(calc)
    energies = dos.get_energies()  # Energy grid relative to the Fermi level
    fermi_level = calc.get_fermi_level()  # obtain the Fermi level from the calculator
    energies = energies - fermi_level     # shift energy grid so that E_F = 0

    # Check the number of spin channels. For spin-polarized (nspins == 2),
    # plot DOS for spin up and spin down separately.
    nspins = calc.get_number_of_spins()
    plt.figure(figsize=(8,5))
    if nspins == 2:
        # For spin-polarized calculations, get DOS for each spin channel.
        dos_up = dos.raw_dos(energies, spin=0, width=width)
        dos_dn = dos.raw_dos(energies, spin=1, width=width)

        # Plot the DOS.
        plt.plot(energies, dos_up, label='Spin Up', color='red')
        plt.plot(energies, dos_dn, label='Spin Down', color='blue')
        plt.legend()

        # Numerically integrate DOS for spin up and spin down.
        # We assume that energies are in eV and that the DOS is in states/eV.
        # Integration from the minimum energy up to 0 
        # (the Fermi level, since energies are relative)
        mask = energies <= 0
        N_up = np.trapezoid(dos_up[mask], energies[mask])
        N_dn = np.trapezoid(dos_dn[mask], energies[mask])
        delta_N = N_up - N_dn

        print("Integrated electron count up to Fermi level:")
        print(f"  Spin up:   {N_up:.3f} electrons")
        print(f"  Spin down: {N_dn:.3f} electrons")
        print(f"Net spin polarization: {delta_N:.3f} electrons")
        print(f"Estimated net magnetic moment: {delta_N:.3f} μB per unit cell")
    else:
        # For non-spin-polarized calculation, there is only one DOS.
        dos_total = dos.raw_dos(energies, width=width)
        plt.plot(energies, dos_total, label='DOS', color='green')
        plt.legend()
        print("Non-spin-polarized calculation; net magnetic moment is 0.")

    # Label the axes and add title
    plt.xlabel(r"$\epsilon - \epsilon_F \ (eV)$")
    plt.ylabel("DOS (states/eV)")
    plt.title(f"Density of States from {filename}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./DOS_{filename.replace(".gpw","")}.png")
    plt.show()

if __name__ == "__main__": main()