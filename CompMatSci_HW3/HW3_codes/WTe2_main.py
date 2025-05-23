import numpy as np
import matplotlib.pyplot as plt
from ase.io import write

from WTe2_structure import build_wte2_structure
from WTe2_search import kpoint_convergence_test, ecut_convergence_test, optimize_lattice_constants, setup_gpaw_paths
from WTe2_bulk import calculate_bulk_modulus

setup_gpaw_paths()

print("Starting crystal structure analysis of 2H-WTe2...")

# Initial structure with literature values
initial_wte2 = build_wte2_structure()
write('initial_wte2.cif', initial_wte2)
print("Initial structure created and saved.")

# Step 1: K-point convergence test with threshold-based detection
print("\n=== Starting k-point convergence test ===")
kpt_results, optimal_kpts = kpoint_convergence_test(
                                                        initial_wte2, 
                                                        kmin=4, 
                                                        kmax=12, 
                                                        kstep=2, 
                                                        ecut=400, 
                                                        threshold=1e-3
                                                    )

# Plot k-point convergence
plt.figure(figsize=(10, 6))
plt.plot(kpt_results['kpoints'], kpt_results['energies'], 'o-')
plt.xlabel(r'K-points ($n$ value in $n \times n \times n_z$ grid)')
plt.ylabel('Energy (eV)')
plt.title('K-point Convergence Test')
plt.grid(True)
plt.savefig('kpoint_convergence.png')

print(f"Optimal k-point grid determined: {optimal_kpts}")

# Step 2: Plane wave cutoff convergence test
print("\n=== Starting plane wave cutoff convergence test ===")
ecut_results, optimal_ecut = ecut_convergence_test(
                                                        initial_wte2, 
                                                        optimal_kpts, 
                                                        ecut_min=300, 
                                                        ecut_max=800, 
                                                        ecut_step=50,
                                                        threshold=1e-3
                                                    )

# Plot cutoff energy convergence
plt.figure(figsize=(10, 6))
plt.plot(ecut_results['ecuts'], ecut_results['energies'], 'o-')
plt.xlabel('Plane Wave Cutoff Energy (eV)')
plt.ylabel('Energy (eV)')
plt.title('Plane Wave Cutoff Convergence Test')
plt.grid(True)
plt.savefig('ecut_convergence.png')

print(f"Optimal plane wave cutoff energy determined: {optimal_ecut} eV")

# Step 3: Lattice constant optimization
print("\n=== Starting lattice constant optimization ===")
# Starting with literature values and varying by ±5%
p_a, p_c = 0.05, 0.05
a_range = (3.496*(1-p_a), 3.496*(1+p_a), 0.05)  # ±p_a*100% around 3.496 Å
c_range = (3.496*(1-p_c), 3.496*(1+p_c), 0.2)  # ±p_c*100% around 14.07 Å

lattice_results, (opt_a, opt_c) = optimize_lattice_constants(
                                                                a_range, 
                                                                c_range, 
                                                                optimal_kpts, 
                                                                optimal_ecut
                                                            )

# Create optimized structure
print(opt_a, opt_c)
opt_wte2 = build_wte2_structure(a=opt_a, c=opt_c)
write('optimized_wte2_final.cif', opt_wte2)
print(f"Optimized structure saved with a = {opt_a:.4f} Å, c = {opt_c:.4f} Å")

# Create surface plots for the energy landscape
a_unique = np.unique(lattice_results['a_values'])
c_unique = np.unique(lattice_results['c_values'])

if len(a_unique) > 1 and len(c_unique) > 1:
    energy_grid = np.zeros((len(a_unique), len(c_unique)))
    
    for i, a in enumerate(a_unique):
        for j, c in enumerate(c_unique):
            # Find the index in the flattened results
            idx = [k for k in range(len(lattice_results['a_values'])) 
                    if abs(lattice_results['a_values'][k] - a) < 1e-6 and 
                        abs(lattice_results['c_values'][k] - c) < 1e-6]
            
            if idx:
                energy_grid[i, j] = lattice_results['energies'][idx[0]]
    
    # Create a surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    A, C = np.meshgrid(a_unique, c_unique, indexing='ij')
    surf = ax.plot_surface(A, C, energy_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Lattice constant a (Å)')
    ax.set_ylabel('Lattice constant c (Å)')
    ax.set_zlabel('Energy (eV)')
    ax.set_title('Energy landscape for lattice parameters optimization')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    plt.savefig('lattice_energy_landscape.png')


# Step 4: Calculate bulk modulus
# TMDs like MoS2 often use Birch-Murnaghan, 
# but Roset-Vinet is favored for covalent-dominated compression
print("\n=== Calculating bulk modulus ===")
for isothermal in {"murnaghan", "birchmurnaghan", "vinet"}:
    B_GPa, v0, e0 = calculate_bulk_modulus(
                                            opt_wte2, 
                                            optimal_kpts, 
                                            optimal_ecut,
                                            scale_range=0.94,
                                            scale_steps=9,
                                            isothermal=isothermal
                                           )

# Calculate volume, lattice parameters for comparison
initial_volume = initial_wte2.get_volume()
optimized_volume = opt_wte2.get_volume()
volume_change = (optimized_volume - initial_volume) / initial_volume * 100

# Summarize results and compare with experiment
# # Literature values for comparison
# Publications from the Terrones group (Penn State University)
# Journal articles in Physical Review B, Advanced Materials, or Nature Materials on WTe2 polymorphs
exp_a = 3.496  # Å
exp_c = 14.07  # Å
exp_B = 65.0  # GPa (literature values differ)

print("\n=== Results Summary ===")
print(f"Initial lattice constants: a = {3.496:.4f} Å, c = {14.07:.4f} Å")
print(f"Optimized lattice constant a: {opt_a:.4f} Å (Literature: {exp_a:.4f} Å)")
print(f"Difference in a: {((opt_a - exp_a) / exp_a * 100):.2f}%")
print(f"Optimized lattice constant c: {opt_c:.4f} Å (Literature: {exp_c:.4f} Å)")
print(f"Difference in c: {((opt_c - exp_c) / exp_c * 100):.2f}%")
print(f"Volume change from initial structure: {volume_change:.2f}%")
print(f"Calculated bulk modulus: {B_GPa:.2f} GPa (Literature: {exp_B:.2f} GPa)")
print(f"Difference in bulk modulus: {((B_GPa - exp_B) / exp_B * 100):.2f}%")
print(f"Equilibrium volume: {v0:.3f} Å³")
print(f"Minimum energy: {e0:.6f} eV")

print("\nOptimal computational parameters:")
print(f"K-point grid: {optimal_kpts}")
print(f"Plane wave cutoff energy: {optimal_ecut} eV")