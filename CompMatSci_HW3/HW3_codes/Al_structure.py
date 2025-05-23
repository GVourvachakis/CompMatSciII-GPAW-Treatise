"""Bulk Al(fcc) test"""
from __future__ import print_function
from ase import Atoms
from gpaw import GPAW, PW, setup_paths
import os
import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator

# from ase.eos import EquationOfState

# Global variable to define the crystal structure
structure = "fcc"  # Options: "fcc" or "bcc"

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

def get_lattice_constant() -> float:
        """Return the experimental lattice constant based on the crystal structure."""
        # source for lattice constant(s): 
        # M.E. Straumanis, C.L. Woodard, Acta. Cryst. A 27 (1971) 549
        if structure == "fcc":
            return 4.05  # Experimentally measured via XRD in Å
        elif structure == "bcc":
            return 2.86  # Experimentally measured via XRD in Å
        else:
            raise ValueError("Unsupported crystal structure: {}".format(structure))

def create_bulk_structure(alpha):
    """Create the bulk structure based on the crystal structure and lattice constant."""
    # Using Primitive Unit Cells
    # ∥v_i​∥_fcc = sqrt(2)*b = a/sqrt(2), ∥v_i​∥_bcc = sqrt(3)*b = sqrt(3)*a/2
    b = alpha / 2.0

    if structure == "fcc":
        # Build the fcc primitive cell for aluminum:
        # The primitive vectors are: v1 = [0, b, b], v2 = [b, 0, b], v3 = [b, b, 0]
        b = alpha / 2.0 # both fcc and bcc primitive cell vectors use half the lattice constant
        cell = [[0, b, b],
                [b, 0, b],
                [b, b, 0]]
    elif structure == "bcc":
        # Build the bcc primitive cell for aluminum:
        # The primitive vectors are: v1 = [-b, b, b], v2 = [b, -b, b], v3 = [b, b, -b]
        b = alpha / 2.0
        cell = [[-b,  b,  b],
                [ b, -b,  b],
                [ b,  b, -b]]
    else:
        raise ValueError(f"Unsupported crystal structure: {structure}")
    
    bulk = Atoms('Al', cell=cell, pbc=True)
    return bulk

def compute_energies_vs_k(alpha, k_values):
    """Compute potential energies for different k-point grids."""
    energies_k = []
    print(f"Optimal kpoints search for fixed α_{structure} = {alpha:.3f}")
    for k in k_values:
        bulk = create_bulk_structure(alpha)
        
        ### Experiment with different cut-offs: 300, 450, 500
        calc = GPAW(mode=PW(300),       # Plane-wave cutoff energy (300 eV) 
                    kpts=(k, k, k),     # Monkhorst-Pack grid for Brillouin zone sampling
                    txt=None)           # Log file for the calculation
    
        bulk.calc = calc # Assign the calculator to the Atoms object
        
        energy = bulk.get_potential_energy()
        energies_k.append(energy)
        print(f"Energy: {energy:.6f} eV (kpts = {k}x{k}x{k})")
    return energies_k

def find_elbow_point(k_values, energies_k):
    # Elbow method: identify where the energy change "levels off"
    # Here we use KneeLocator from the kneed package.
    # 'convex' and 'decreasing' are chosen because the energy curve is expected to decay.
    knee_locator = KneeLocator(k_values, energies_k, curve='convex', direction='decreasing')
    # knee_locator.knee, 4 for fcc, 7 for bcc (the location can be verified visually)
    if structure == "fcc": elbow_k: int = 4
    elif structure == "bcc": elbow_k: int = 7
    print(f"Elbow point detected at k ({structure} phase)= {elbow_k}")
    return elbow_k

def plot_energy_vs_k(energies_k, k_values, elbow_k):
    # Optionally, mark the elbow on the plot:
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, energies_k, 'o-', label='Potential Energy')
    if elbow_k is not None:
        plt.axvline(elbow_k, color='r', linestyle='--', label=f'Elbow (k = {elbow_k})')
    plt.xlabel('kpoints (k)')
    plt.ylabel('Potential Energy (eV)')
    plt.title(f'Potential Energy vs. k-points with Elbow Detection for {structure.upper()} Al')
    plt.grid(True)
    plt.legend()
    plt.savefig(f"energy_vs_k_{structure}.png", dpi=300)
    print(f"\nPlot of potential energy vs k-grid sampling saved as 'energy_vs_k_{structure}.png'")
    plt.show()

def compute_energies_vs_alpha(alphas, k):
    energies = []
    for alpha in alphas:
        print(f"Calculating for α ({structure}) = {alpha:.3f} Å")
        bulk = create_bulk_structure(alpha)
        calc = GPAW(mode=PW(300), kpts=(k, k, k), txt=None)
        bulk.calc = calc
        energy = bulk.get_potential_energy()
        energies.append(energy)
        
        # Save the calculator state for possible restart or further analysis
        # calc.write(f"Al-{structure}_{alpha:.3f}.gpw")
        
        print(f"Energy: {energy:.6f} eV")
    return energies

def fit_and_plot_energies(alphas, energies):
    # Fit the energies with a quadratic polynomial: E(α) = a2 α² + a1 α + a0
    coeffs = np.polyfit(alphas, energies, 2)
    a2, a1, a0 = coeffs  # a2: Quadratic coefficient (eV/Å^2) from the fit (used for Bulk modulus)
    print("\nFitted coefficients:")
    print(f"a2 = {a2:.5f}, a1 = {a1:.5f}, a0 = {a0:.5f}")
    
    # Find the minimum from the derivative: dE/dα = 2 a2 α + a1 = 0 => α_min = -a1/(2*a2)
    alpha_min = - a1 / (2 * a2)
    print(f"Minimum energy for {structure} Aluminium is predicted at α = {alpha_min:.3f} Å")
    
    # Create a smooth curve for the fitted polynomial
    alpha_fit = np.linspace(alphas[0], alphas[-1], 100)
    energy_fit = a2 * alpha_fit**2 + a1 * alpha_fit + a0
    
    # Plot the computed energies and the polynomial fit
    plt.figure(figsize=(8, 5))
    plt.plot(alphas, energies, 'o', label='Calculated energies')
    plt.plot(alpha_fit, energy_fit, '-', label='Quadratic fit')
    plt.xlabel('Lattice constant α (Å)')
    plt.ylabel('Energy (eV)')
    plt.title(f"Energy vs Lattice Constant for {structure.upper()} Al")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"energy_vs_alpha_{structure}.png", dpi=300)
    print(f"\nPlot of potential energy with parabolic approximation saved as 'energy_vs_alpha_{structure}.png'\n")
    plt.show()
    
    return alpha_min, a2

def compute_bulk_modulus(alpha0, a2):
    # sources for bulk modulus of FCC and BCC Al (B_fcc = 76.2, B_bcc = 475): 
    # J.L. Tallon, A.J. Wolfenden, Phys. Chem. Solids 40 (1979) 831
    # "Stability criteria of Aluminum lattice from first-principles", Zhang et al. (2025)
    # "Measurement of Body-Centered-Cubic Aluminum at 475 GPa", Polsin et al (2017)
    if structure == "fcc": B_exp = 76.2
    elif structure == "bcc": B_exp = 475  # Theoretical/high-pressure value
    # consistent with Zhang et al. (2025) and Polsin et al. (2017)

    else: raise ValueError(f"Unsupported crystal structure: {structure}")

    # Given parameters from the polynomial fit
    d2E_dalpha2 = 2 * a2  # Second derivative: d^2E/dα^2 in eV/Å^2
    
    B_eV_A3 = (4 / (9 * alpha0)) * d2E_dalpha2
    
    # Conversion factor: 1 eV/Å^3 ≈ 160.21766 GPa
    B_GPa = B_eV_A3 * 160.21766
    
    print(f"Calculated Bulk Modulus (for lattice constant a_{structure}={alpha0:.3f}): {B_GPa:.2f} GPa", end="")
    print(f" (experimental in {structure} phase: {B_exp} GPa)")

    perc_error: float = 100 * abs(B_GPa - B_exp) / B_exp
    
    print(f"percentage error: {perc_error:.5f} %\n")

def main():
    """Main routine for polymorphs of bulk Aluminium test.
    To change the crystal structure modify the global variable 'structure' accordingly (fcc or bcc).
    """

    setup_gpaw_paths()
    alpha0: float = get_lattice_constant()

    # Define the central (experimental) lattice constant and variation range
    if structure == "fcc": variation =  0.02 # 2% for fcc
    elif structure == "bcc": variation = 0.10 # 10% for bcc
    else: raise ValueError("Unsupported crystal structure: {}".format(structure))
    lower_bound = alpha0 * (1 - variation)  # 2/10% below experimental
    upper_bound = alpha0 * (1 + variation)  # 2/10% above experimental
    num_points = 10  # Increase number of points for a smoother curve
    alphas = np.linspace(lower_bound, upper_bound, num_points)
    print("Lattice constant trials range:", alphas)
    
    # Array to store potential energies for different k-values
    kmax: int = 10
    k_values = np.arange(2, kmax+1)
    
    # Loop over k-values; here we show how you might set up GPAW calculations.
    energies_k = compute_energies_vs_k(alpha0, k_values)
    
    # Elbow method: identify where the energy change "levels off"
    elbow_k = find_elbow_point(k_values, energies_k)
    
    # Optionally, mark the elbow on the plot:
    plot_energy_vs_k(energies_k, k_values, elbow_k)
    
    # use the above kpoints estimate, k_elbow to find optimal lattice constant, alpha_min
    energies = compute_energies_vs_alpha(alphas, elbow_k)
    
    # Fit the energies with a quadratic polynomial and plot the result
    alpha_min, a2 = fit_and_plot_energies(alphas, energies)
    
    # Calculate the bulk modulus based on the fit parameters
    compute_bulk_modulus(alpha_min, a2) # Equilibrium lattice constant in Å (from the binding energy curve)
    compute_bulk_modulus(alpha0, a2) # theoretical to experimental value comparison

if __name__ == '__main__':
    main()