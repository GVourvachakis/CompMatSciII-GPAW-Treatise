#!/usr/bin/env python
# surface_DFT.py - Calculate surface energies using DFT with and without relaxation 
# and compare with the Dangling Bond Model (DBM).
# For dynamic parallelization with user-defined available cores, run: 
# `mpiexec -np {available_cores} python surface_DFT.py`
"""
Compute surface energy, γ, from DFT and DFT + relaxation.

Expected Results:

1. The relaxed surface energies should be lower than the unrelaxed ones due to structural optimization.

2. _(111) should have the lowest surface energy (most stable surface).
   _(110) and _(211) should have higher surface energies (true for FCC phase).

3. DFT results might differ significantly from the dangling bond model, 
   especially for more complex surfaces like (211).
"""

from ase.build import bulk
from gpaw import GPAW, setup_paths
from ase.optimize import BFGS
import os
import matplotlib.pyplot as plt
import numpy as np
from dbm import surface_energy_dbm 
from utils_DFT import create_dir, create_slab, get_kpoints, save_layers, save_results, load_results,\
                      apply_constraints, plot_layer_results, get_parallel_config, ols_surface_energy

# Set up GPAW dataset path
setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
setup_paths[:] = [setup_path]  # Replace all existing paths
os.environ['GPAW_SETUP_PATH'] = setup_path

material: str = "Pd"

# Then use this in your calculator initialization:
def calculate_bulk_energy():
    """Calculate the energy of bulk Al/Pd."""
    print(f"Calculating bulk energy {material}...")
    if material == "Al": a=4.05; magnetic=False; xc="LDA" # Experimentally measured via XRD in Å
    elif material == "Pd": a=3.89; magnetic=True; xc="PBE" # -//-
    else: raise ValueError(f"Unsupported fcc material: {material}")

    bulk_material = bulk(material,'fcc',a)

    # Set k-points for bulk
    kpts = (9, 9, 9)
    
    # Get appropriate parallel configuration
    parallel_config = get_parallel_config(kpts)
    print(f"Using parallel configuration for bulk: {parallel_config}")

    # Enable symmetry: using time-reversal and point-group symmetry
    calc = GPAW(h=0.25, kpts=kpts, xc=xc, 
                symmetry={'time_reversal': True, 'point_group': True},
                convergence={'energy': 1e-6, 'density': 1e-6},
                spinpol=magnetic,         # Include spin polarization 
                # in Pd since can exhibit magnetic behavior, especially at surfaces
                parallel=parallel_config)  # Use dynamic configuration

    bulk_material.calc = calc
    bulk_energy = bulk_material.get_potential_energy()
    n_atoms_bulk = len(bulk_material)
    return bulk_energy / n_atoms_bulk

def surface_energy_unrelaxed(surface_type, bulk_energy_per_atom):
    """Calculate unrelaxed surface energy for the specified surface."""
    print(f"\nCalculating unrelaxed surface energy for {material}({surface_type})...")
    
    # Create slab
    slab = create_slab(surface_type)
    kpts = get_kpoints(surface_type)
    
    # Save information
    n_atoms = len(slab)
    area = slab.cell[0][0] * slab.cell[1][1] - slab.cell[0][1] * slab.cell[1][0]
    
    # Get parallel configuration
    parallel_config = get_parallel_config(kpts)
    print(f"Using parallel configuration: {parallel_config}")
    
    # Setup calculator
    if material == 'Al': xc='LDA'; magnetic=False # simpler post-transition metal
    elif material == 'Pd': xc='PBE'; magnetic=True # transition metal, with d-orbital electrons
    else: raise ValueError(f"Unsupported fcc material: {material}")

    calc = GPAW(h=0.25, kpts=kpts, xc=xc, spinpol=magnetic, parallel=parallel_config)
    slab.calc = calc
    
    # Calculate energy
    slab_energy = slab.get_potential_energy()
    
    # Calculate surface energy (E_slab - n*E_bulk)/(2*A)
    # Factor of 2 because we have two surfaces (top and bottom)
    surface_energy = (slab_energy - n_atoms * bulk_energy_per_atom) / (2 * area)
    
    # Convert from eV/Å² to J/m²
    surface_energy_Jm2 = surface_energy * 16.022  # 1 eV/Å² = 16.022 J/m²
    
    print(f"{material}({surface_type}) unrelaxed: E_slab = {slab_energy:.4f} eV, Area = {area:.2f} Å², γ = {surface_energy_Jm2:.2f} J/m²")
    
    return {
        'slab_energy': slab_energy,
        'n_atoms': n_atoms,
        'area': area,
        'surface_energy': surface_energy,
        'surface_energy_Jm2': surface_energy_Jm2
    }

def surface_energy_relaxed(surface_type, bulk_energy_per_atom):
    """Calculate relaxed surface energy for the specified surface."""
    print(f"\nCalculating relaxed surface energy for {material}({surface_type})...")
    
    # Create slab
    slab = create_slab(surface_type)
    kpts = get_kpoints(surface_type)
    
    # Save information
    n_atoms = len(slab)
    area = slab.cell[0][0] * slab.cell[1][1] - slab.cell[0][1] * slab.cell[1][0]
    
    # Get parallel configuration
    parallel_config = get_parallel_config(kpts)
    print(f"Using parallel configuration: {parallel_config}")

    # Setup calculator (reduced grid spacing and stricter convergence criteria for electronic steps)
    if material == 'Al': xc='LDA'; magnetic=False # post-transition metal, p-orbital electrons
    elif material == 'Pd': xc='PBE'; magnetic=True # transition metal, d-orbital electrons
    else: raise ValueError(f"Unsupported fcc material: {material}")
    calc = GPAW(h=0.25, kpts=kpts, xc=xc, # if resources permit it: h = 0.18, xc='PBE' for Al
                spinpol=magnetic,
                parallel=parallel_config)

    slab.calc = calc

    print(f"\nInitial energy: {slab.get_potential_energy()}")
    # Relax structure
    trajectory_file = f'slab{material+surface_type}.traj'
    dyn = BFGS(slab, trajectory=trajectory_file)
    # tighter force tolerance and increase max iterations for better convergence
    dyn.run(fmax=0.1) #, steps=200)
    
    print(f"\nFinal energy: {slab.get_potential_energy()}")
    print(f"\nRelaxation steps: {dyn.get_number_of_steps()}\n")

    # Get relaxed energy
    slab_energy = slab.get_potential_energy()
    
    # Calculate surface energy (E_slab - n*E_bulk)/(2*A)
    surface_energy = (slab_energy - n_atoms * bulk_energy_per_atom) / (2 * area)
    
    # Convert from eV/Å² to J/m²
    surface_energy_Jm2 = surface_energy * 16.022  # 1 eV/Å² = 16.022 J/m²
    
    print(f"{material}({surface_type}) relaxed: E_slab = {slab_energy:.4f} eV, Area = {area:.2f} Å², γ = {surface_energy_Jm2:.2f} J/m²")
    
    return {
        'slab_energy': slab_energy,
        'n_atoms': n_atoms,
        'area': area,
        'surface_energy': surface_energy,
        'surface_energy_Jm2': surface_energy_Jm2
    }

def layer_analysis(surface_type, bulk_energy_per_atom, layers_to_test=[5, 6, 7, 8]):
    """
    Calculate surface energies for different numbers of layers and create plots.
    Apply linear fitting to estimate surface energy using E = N_l*E_bulk + 2γA,
    where N_l is the number of layers.
    
    Args:
        surface_type (str): The surface type ('100', '110', '111', or '211')
        bulk_energy_per_atom (float): The calculated bulk energy per atom
        layers_to_test (list): List of layer counts to test
    """
    
    print(f"\nPerforming layer analysis for {material}({surface_type})...")
    
    # Store results
    results = {
        'layers': [],
        'energies': [],
        'surface_energies': [],
        'surface_energies_Jm2': [],
        'relaxation_steps': [],
        'max_displacements': [],
        'n_atoms': [],
        'areas': []
    }
    
    for n_layers in layers_to_test:
        print(f"\nCalculating for {n_layers} layers...")
        
        # Create slab with specified number of layers
        slab = create_slab(surface_type, layers=n_layers)
        kpts = get_kpoints(surface_type)
        
        # Save information
        n_atoms = len(slab)
        area = slab.cell[0][0] * slab.cell[1][1] - slab.cell[0][1] * slab.cell[1][0]
        #atoms_per_layer = n_atoms / n_layers
        
        # Get parallel configuration
        parallel_config = get_parallel_config(kpts)
        print(f"Using parallel configuration for relaxed DFT: {parallel_config}")

        # Setup calculator with improved parameters
        if material == 'Al': xc='LDA'; magnetic=False # simpler metal
        elif material == 'Pd': xc='PBE'; magnetic=True # transition metal, with d-orbital electrons
        else: raise ValueError(f"Unsupported fcc material: {material}")

        calc = GPAW(h=0.25, kpts=kpts, xc=xc, # if resources permit, choose h=0.18
                   #convergence={'energy': 1e-6, 'density': 1e-6},
                   symmetry={'time_reversal': True, 'point_group': True}, # enforce symmetry during the relaxation process
                   spinpol=magnetic,
                   parallel=parallel_config)        
        
        slab.calc = calc

        # Fix bottom layers (approximately half of the slab)
        # apply_constraints(slab, n_layers, n_atoms)
        
        # for atom in slab:
        #     print(atom.index, atom.position)

        # Store initial positions and energy
        positions_before = slab.get_positions().copy()
        initial_energy = slab.get_potential_energy()

        # Relax structure with tighter convergence criteria
        trajectory_file = f'slab_{material}({surface_type})__{n_layers}layers.traj'
        dyn = BFGS(slab, trajectory=trajectory_file)
        dyn.run(fmax=0.1)#, steps=100)
        
        # Get relaxed energy and number of steps
        slab_energy = slab.get_potential_energy()
        relaxation_steps = dyn.get_number_of_steps()
        
        # Calculate maximum displacement during relaxation
        positions_after = slab.get_positions()
        displacements = np.linalg.norm(positions_after - positions_before, axis=1)
        max_displacement = np.max(displacements)
        
        # Calculate surface energy (E_slab - n*E_bulk)/(2*A)
        surface_energy = (slab_energy - n_atoms * bulk_energy_per_atom) / (2 * area)
        
        # Convert from eV/Å² to J/m²
        surface_energy_Jm2 = surface_energy * 16.022  # 1 eV/Å² = 16.022 J/m²
        
        print(f"Al({surface_type}) {n_layers} layers: E_slab = {slab_energy:.4f} eV (initial: {initial_energy:.4f} eV)")
        print(f"Area = {area:.2f} Å², γ = {surface_energy_Jm2:.2f} J/m²")
        print(f"Relaxation steps: {relaxation_steps}, Max displacement: {max_displacement:.4f} Å")
        
        # Store results
        results['layers'].append(n_layers)
        results['energies'].append(slab_energy)
        results['surface_energies'].append(surface_energy)
        results['surface_energies_Jm2'].append(surface_energy_Jm2)
        results['relaxation_steps'].append(relaxation_steps)
        results['max_displacements'].append(max_displacement)
        results['n_atoms'].append(n_atoms)
        results['areas'].append(area)

    # Perform linear fitting to estimate surface energy (OLS estimator)
    ols_surface_energy(results, surface_type, bulk_energy_per_atom)

    # Generate and save plots
    plot_layer_results(results, surface_type)

    # Save numerical results
    save_layers(results, surface_type)
    
    return results

def compare_with_dangling_bond_model(dft_results, dbm_results):
    """Compare DFT results with the dangling bond model."""
    print("\nComparison between DFT and Dangling Bond Model:")
    print("Surface | DFT Unrelaxed (J/m²) | DFT Relaxed (J/m²) | DBM (J/m²)")
    print("--------|---------------------|-------------------|------------")
    
    for surface in ['100', '110', '111', '211']:
        dft_unrelaxed = dft_results['unrelaxed'][surface]['surface_energy_Jm2']
        dft_relaxed = dft_results['relaxed'][surface]['surface_energy_Jm2']
        dbm = dbm_results[surface]
        print(f"{material}({surface}) | {dft_unrelaxed:.2f} | {dft_relaxed:.2f} | {dbm:.2f}")

def main():
    """Main function to calculate all surface energies with layer analysis."""

    #----------Constants----------#
    z_b: int = 12  # Bulk coordination number for FCC (both Al and Pd)
    
    # enthalpy of atomisation (from webelements)
    if material == 'Al': ΔatH: float = 326 # kJ mol^{-1}
    elif material == 'Pd': ΔatH: float = 377 # kJ mol^{-1}
    else: raise ValueError(f"Unsupported fcc material: {material}")
    
    Avogadro: float = 6.02214 * 1e23
    
    E_at: float = (ΔatH/Avogadro)*1e3 # J/atom
    # print(f"\nAtomization energy of {material}:\t{E_at} J/atom\n")
    J_to_Jm2: float = 1e20  # Convert J/Å² to J/m²
    vacuum: float = 15.0 # Increased vacuum for better isolation
    layers: int = 5
    
    surfaces = ['100','110','111','211']

    # # Create results directory if it doesn't exist
    create_dir('surface_results')
    # # Calculate bulk energy first (once)
    bulk_energy_per_atom = calculate_bulk_energy()
    print(f"Bulk energy per atom for {material}: {bulk_energy_per_atom:.4f} eV")
    # # Bulk energy per atom: -3.7130 eV for kpts = (12x12x12)
    # # Bulk energy per atom: -3.6850 eV for kpts = (9x9x9), xc = 'PBE'
    # # Bulk energy per atom: parallelized: -3.7350 eV for kpts = (12x12x12)
    # # Bulk energy per atom: -4.1397 eV for kpts = (9x9x9), xc = 'LDA'
    # # Bulk energy per atom for Pd: -4.586717033689929 eV for kpts = (9x9x9), xc = 'PBE'
    # # Store all results

    results = {
        'bulk_energy_per_atom': bulk_energy_per_atom,
        'unrelaxed': {},
        'relaxed': {},
        'layer_analysis': {}
    }

    # Load existing results
    #results = load_results(filename=f'surface_energy_results_{material}.pkl')
    
    # Calculate relaxed surface energies
    if 'relaxed' not in list(results.keys()):
        results['relaxed'] = {}
    # for surface in surfaces:
    #     results['unrelaxed'][surface] = surface_energy_unrelaxed(surface, bulk_energy_per_atom)

    for surface in surfaces:
        results['relaxed'][surface] = surface_energy_relaxed(surface, bulk_energy_per_atom)

    # Save updated results
    save_results(results, f'./surface_results/surface_energy_results_{material}.pkl')
    
    # In case you already run the above analysis, unpickle the saved instance as hashmap
    # results = load_results(f'./surface_results/surface_energy_results_{material}.pkl')
    # print(results)
    
    # Print summary
    print(f"\nSummary of Surface Energies (J/m²) for {material}:")
    print("Surface | Unrelaxed | Relaxed | Difference")
    print("--------|-----------|---------|------------")
    surfaces = ['100','110','111','211']
    for surface in surfaces:
        unrelaxed = results['unrelaxed'][surface]['surface_energy_Jm2']
        relaxed = results['relaxed'][surface]['surface_energy_Jm2']
        diff = unrelaxed - relaxed
        diff_percent = (diff / unrelaxed) * 100
        
        print(f"{material}({surface}) | {unrelaxed:.2f} | {relaxed:.2f} | {diff:.2f} ({diff_percent:.1f}%)")
    
    
    # # ------------ Dangling Bond Model Construction and Comparison ---------- #
    # Create slabs dictionary
    slabs = {}
    # Calculate surface energies and store in dbm_results dictionary
    dbm_results = {}

    for system in surfaces:
        # Create and store each slab with the corresponding surface type
        slabs[system] = create_slab(surface_type=system, layers=layers, vacuum=vacuum)
        gamma, _, _ = surface_energy_dbm(slabs[system], z_b, E_at, J_to_Jm2, system)
        dbm_results[system] = gamma

    # Compare with the dangling bond model
    compare_with_dangling_bond_model(results, dbm_results)

    # ----------------------Layer Analysis-------------------------#    
    # Perform layer analysis for each surface
    surfaces = ['100','110','111', '211']

    for surface in surfaces:
        print(f"\n{'='*50}")
        print(f"Starting layer analysis for {material}({surface})")
        print(f"{'='*50}")
        results['layer_analysis'][surface] = layer_analysis(surface, bulk_energy_per_atom,\
                                                            layers_to_test=[5,6,7,8])
    
    # Create a combined plot of all surfaces
    plt.figure(figsize=(10, 6))
    for surface in surfaces:
        layer_data = results['layer_analysis'][surface]
        plt.plot(layer_data['layers'], layer_data['surface_energies_Jm2'], 'o-', 
                 linewidth=2, markersize=8, label=f'{material}({surface})')
    
    # Add theoretical values as horizontal lines
    
    if material == 'Al':
        theoretical_values = {'100': 2.21, '110': 1.95, '111': 1.91, '211': 2.25}
    elif material == 'Pd': 
        theoretical_values = {'100': 2.55, '110': 2.25, '111': 2.21, '211': 2.60} 
    else: raise ValueError(f"Unsupported fcc material: {material}")

    for surface, value in theoretical_values.items():
        plt.axhline(y=value, linestyle='--', alpha=0.5, 
                   label=f'{material}({surface}) theoretical')

    plt.xlabel('Number of Layers', fontsize=14)
    plt.ylabel('Surface Energy (J/m²)', fontsize=14)
    plt.title(f'Surface Energy vs. Number of Layers for Different {material} Surfaces', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(f'surface_results/figures/all_surfaces_comparison_{material}.png', dpi=300)

    # update results with the layer analysis
    save_results(results, f'surface_results/surface_energy_results_{material}.pkl')
    
    print(list(results.keys()))

if __name__ == "__main__": main()