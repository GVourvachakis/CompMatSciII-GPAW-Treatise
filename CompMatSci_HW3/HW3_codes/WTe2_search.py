import os
import time
import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from ase.constraints import FixSymmetry
from ase.optimize import BFGS
from gpaw import GPAW, PW, setup_paths

# from dftd3.ase import DFTD3
# from ase.calculators.mixing import SumCalculator

# from gpaw.analyse.hirshfeld import HirshfeldPartitioning
# from gpaw.analyse.vdwradii import vdWradii
# from ase.calculators.vdwcorrection import vdWTkatchenko09prl

# from ase.calculators.vdwcorrection import vdw_correction_factory <- DEPRECATED
# from gpaw.vdw import vdw_correction_factory <- Not compatible with GPAW 25.1.0

from WTe2_structure import build_wte2_structure

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

# Step 2: K-point convergence test with threshold-based detection
def kpoint_convergence_test(atoms, kmin=2, kmax=12, kstep=2, ecut=400, threshold=1e-3):
    """
    Perform k-point convergence test with threshold-based convergence detection.

    Args:
        atoms: ASE Atoms object
        kmin: Minimum k-points in each direction
        kmax: Maximum k-points in each direction
        kstep: Step size for k-points
        ecut: Plane wave cutoff energy in eV
        threshold: Energy convergence threshold in eV

    Returns:
        Dictionary with k-points and corresponding energies, and optimal k-points
    """
    results = {'kpoints': [], 'kpts_grid': [], 'energies': []}

    # Add symmetry constraint to preserve hexagonal structure
    atoms_copy = atoms.copy()
    atoms_copy.set_constraint(FixSymmetry(atoms_copy))
    
    # Determine aspect ratio for anisotropic k-point sampling
    a_len = np.linalg.norm(atoms.cell[0])
    c_len = np.linalg.norm(atoms.cell[2])
    k_ratio = a_len / c_len
    
    for k in range(kmin, kmax+1, kstep):
        # For this layered hexagonal system, use anisotropic k-point sampling
        # Determine k-points in z-direction based on aspect ratio
        kz = max(1, int(k * k_ratio + 0.5))
        kpts = [k, k, kz]
        

        # Create GPAW calculator using GPU-optimized settings
        calc = GPAW(mode=PW(ecut),
                    kpts=kpts,
                    txt=f'kpt_conv_test_{k}x{k}x{kz}.txt',
                    xc='PBE',
                    parallel={'sl_auto': True}
                    )

        atoms_copy.calc = calc
        energy = atoms_copy.get_potential_energy()

        results['kpoints'].append(k)
        results['kpts_grid'].append(kpts)
        results['energies'].append(energy)
        print(f"K-points: {kpts}, Energy: {energy} eV")

        # Check for convergence using threshold
        if len(results['energies']) > 1:
            energy_diff = abs(results['energies'][-1] - results['energies'][-2])
            print(f"Energy difference from previous k-point: {energy_diff:.6f} eV")
            if energy_diff < threshold:
                print(f"Converged at k-points {kpts} within threshold {threshold} eV")
                break

    # Determine optimal k-points
    optimal_k = None
    if len(results['energies']) > 1:
        # Find where energy difference falls below threshold
        for i in range(1, len(results['energies'])):
            if abs(results['energies'][i] - results['energies'][i-1]) < threshold:
                optimal_k = results['kpoints'][i]
                break
        
        if optimal_k is None:
            optimal_k = results['kpoints'][-1]  # Fallback to largest value
    else:
        optimal_k = results['kpoints'][0]  # Only one k-point tested
    
    # Determine optimal k-grid
    k_idx = results['kpoints'].index(optimal_k)
    optimal_kpts = results['kpts_grid'][k_idx]
    
    return results, optimal_kpts

# Step 3: Plane wave cutoff convergence test with threshold-based detection
def ecut_convergence_test(atoms, kpts, ecut_min=300, ecut_max=500, ecut_step=50, threshold=1e-3):
    """
    Perform plane wave cutoff energy convergence test with threshold-based detection.

    Args:
        atoms: ASE Atoms object
        kpts: K-point grid to use (determined from previous test)
        ecut_min: Minimum cutoff energy in eV
        ecut_max: Maximum cutoff energy in eV
        ecut_step: Energy step size in eV
        threshold: Energy convergence threshold in eV

    Returns:
        Dictionary with cutoff energies and corresponding energies, and optimal cutoff
    """
    results = {'ecuts': [], 'energies': []}
    
    # Add symmetry constraint to preserve hexagonal structure
    atoms_copy = atoms.copy()
    atoms_copy.set_constraint(FixSymmetry(atoms_copy))

    print(f"ecut in {ecut_min,ecut_max, ecut_step}")

    for ecut in range(ecut_min, ecut_max+1, ecut_step):
        # Create GPAW calculator
        calc = GPAW(mode=PW(ecut),
                    kpts=kpts,
                    txt=f'ecut_conv_test_{ecut}.txt',
                    xc='PBE',
                    parallel={'sl_auto': True}
                    )

        atoms_copy.calc = calc
        energy = atoms_copy.get_potential_energy()

        results['ecuts'].append(ecut)
        results['energies'].append(energy)
        print(f"Cutoff energy: {ecut} eV, Energy: {energy} eV")

        # Check for convergence using threshold
        if len(results['energies']) > 1:
            energy_diff = abs(results['energies'][-1] - results['energies'][-2])
            print(f"Energy difference from previous cutoff: {energy_diff:.6f} eV")
            if energy_diff < threshold:
                print(f"Converged at cutoff {ecut} eV within threshold {threshold} eV")
                break

    # Determine optimal cutoff energy
    optimal_ecut = None
    if len(results['energies']) > 1:
        # Find where energy difference falls below threshold
        for i in range(1, len(results['energies'])):
            if abs(results['energies'][i] - results['energies'][i-1]) < threshold:
                optimal_ecut = results['ecuts'][i]
                break
        
        if optimal_ecut is None:
            optimal_ecut = results['ecuts'][-1]  # Fallback to largest value
    else:
        optimal_ecut = results['ecuts'][0]  # Only one cutoff tested
    
    return results, optimal_ecut


# Step 4: Lattice constant optimization with vdW corrections and symmetry constraints
def optimize_lattice_constants(a_range, c_range, kpts, ecut, vacuum=0.0, max_steps=30, fmax=0.05):
    """
    Perform a grid search for optimizing the lattice constants a and c with 
    van der Waals corrections and symmetry preservation.
    Generates a binding curve of energy vs c/a ratio.
    
    Args:
        a_range: Range of a values to test as (min, max, step)
        c_range: Range of c values to test as (min, max, step)
        kpts: K-point grid to use
        ecut: Plane wave cutoff energy in eV
        vacuum: Vacuum layer along z (for 2D systems)
        max_steps: Maximum number of ionic steps in relaxation
        fmax: Force convergence criterion in eV/Å

    Returns:
        Dictionary with a, c values and corresponding energies and the optimal (a, c)
    """

    # Start timing
    start_time = time.time()

    a_min, a_max, a_step = a_range
    c_min, c_max, c_step = c_range
    # Use fewer points in the grid for faster search
    a_values = np.linspace(a_min, a_max, num=5)  # Reduced number of points
    c_values = np.linspace(c_min, c_max, num=5)  # Reduced number of points

    results = {'a_values': [], 'c_values': [], 'energies': [], 'c_over_a': []}
    min_energy = float('inf')
    opt_a, opt_c = None, None

    # First pass: Coarse grid search
    print("=== First pass: Coarse grid search ===")
    for a in a_values:
        for c in c_values:
            print(f"Testing a = {a:.3f} Å, c = {c:.3f} Å")

            # Generate structure with correct symmetry
            atoms = build_wte2_structure(a=a, c=c, vacuum=vacuum)
            
            # Add symmetry constraint
            atoms.set_constraint(FixSymmetry(atoms))

            # GPAW base calculator
            base_calc = GPAW(mode=PW(ecut * 0.8),  # Reduced cutoff for initial screening
                             kpts=[k//2 + 1 for k in kpts],  # Reduced k-points
                             txt=f'opt_a{a:.3f}_c{c:.3f}.txt',
                             xc='optPBE-vdW',
                             parallel={'sl_auto': True}
                            )
            
            # base_calc = GPAW(mode=PW(ecut),
            #                   kpts=kpts,
            #                   txt=f'refined_opt_a{a:.3f}_c{c:.3f}.txt',
            #                   xc='vdW-DF2',  # Non-local van der Waals functional
            #                   parallel={'sl_auto': True})
                
            # Alternative if vdW-DF2 isn't preferred or available: optPBE-vdW or optB88-vdW
            # base_calc = GPAW(mode=PW(ecut),
            #               kpts=kpts,
            #               txt=f'refined_opt_a{a:.3f}_c{c:.3f}.txt',
            #               xc='optPBE-vdW',  # Alternative vdW functional
            #               parallel={'sl_auto': True})

            #For GPAW 25.1.0 specifically, the following vdW-aware functionals are available:
            # 1. vdW-DF (the original Dion et al. functional)
            # 2. vdW-DF2 (the improved version with better accuracy)
            # 3. optPBE-vdW (optimized for better overall performance)
            # 4. optB88-vdW (another optimized variant)
            # 5. BEEF-vdW (Bayesian Error Estimation Functional with vdW)

            # Create a TS09 vdW correction calculator using GPAW's analysis modules,
            # dispersion correction for better description of layered materials.
            # ts_calc = vdWTkatchenko09prl(
            #                                 HirshfeldPartitioning(base_calc),
            #                                 vdWradii(atoms.get_chemical_symbols(), 'PBE')
            #                             )    
            
            # Dispersion correction on top of your base calculation (GPAW + DFT-D3 vdW correction).
            # (Make sure DFTD3 is installed: pip install dftd3 or conda install -c conda-forge dftd3)
            # calc = SumCalculator([base_calc, DFTD3(method='PBE', damping='bj')])  # Using Becke-Johnson damping
            # You need to specify the damping parameter when creating the DFTD3 calculator. 
            # The most common damping methods for DFT-D3 are "zero" (D3(0)) or "bj" (D3(BJ)).
            # atoms.calc = calc

            atoms.calc = base_calc
    
            try:
                # Skip full ionic relaxation in first pass - just get energy
                energy = atoms.get_potential_energy()
                c_over_a = c / a

                results['a_values'].append(a)
                results['c_values'].append(c)
                results['energies'].append(energy)
                results['c_over_a'].append(c_over_a)

                print(f"Coarse energy: a = {a:.3f} Å, c = {c:.3f} Å, E = {energy:.6f} eV")
                
                if energy < min_energy:
                    min_energy = energy
                    opt_a = a
                    opt_c = c
                    
            except Exception as e:
                print(f"Error calculating for a={a}, c={c}: {str(e)}")
                print("Skipping this configuration and continuing with next...")
                continue
    
    # Find the best point and neighborhood for refinement
    if opt_a is not None and opt_c is not None:
        print(f"\n=== First pass complete ===")
        print(f"Best parameters from coarse search: a = {opt_a:.4f} Å, c = {opt_c:.4f} Å")
        
        # Second pass: Refined search around best point
        print("\n=== Second pass: Refined search around best point ===")
        
        # Define a smaller range around the optimum from first pass
        a_refine_min = max(a_min, opt_a - a_step)
        a_refine_max = min(a_max, opt_a + a_step)
        c_refine_min = max(c_min, opt_c - c_step)
        c_refine_max = min(c_max, opt_c + c_step)
        
        # Create refined grid with 3 points in each dimension
        a_refined = np.linspace(a_refine_min, a_refine_max, 3)
        c_refined = np.linspace(c_refine_min, c_refine_max, 3)
        
        # Clear previous results for refined search
        refined_results = {'a_values': [], 'c_values': [], 'energies': [], 'c_over_a': []}
        min_energy = float('inf')
        
        for a in a_refined:
            for c in c_refined:
                print(f"Refining a = {a:.3f} Å, c = {c:.3f} Å")
                
                # Generate structure with correct symmetry
                atoms = build_wte2_structure(a=a, c=c, vacuum=vacuum)
                
                # Add symmetry constraint
                atoms.set_constraint(FixSymmetry(atoms))
                
                # GPAW calculator with full accuracy
                base_calc = GPAW(mode=PW(ecut),
                              kpts=kpts,
                              txt=f'refined_opt_a{a:.3f}_c{c:.3f}.txt',
                              xc='vdW-DF2',
                              parallel={'sl_auto': True})
                
                atoms.calc = base_calc
                
                try:
                    energy = atoms.get_potential_energy()
                    print(f"Initial energy: {energy:.6f} eV")
                    
                    # Ionic relaxation with limited steps
                    dyn = BFGS(atoms, logfile=f'relax_a{a:.3f}_c{c:.3f}.log')
                    dyn.run(fmax=fmax, steps=max_steps)  # Limit relaxation steps
                    
                    energy = atoms.get_potential_energy()
                    c_over_a = c / a
                    
                    refined_results['a_values'].append(a)
                    refined_results['c_values'].append(c)
                    refined_results['energies'].append(energy)
                    refined_results['c_over_a'].append(c_over_a)
                    
                    print(f"Results after relaxation: a = {a:.3f} Å, c = {c:.3f} Å, E = {energy:.6f} eV")
                    
                    if energy < min_energy:
                        min_energy = energy
                        opt_a = a
                        opt_c = c
                    
                    # Save optimized structure only for the best candidates
                    if energy <= min_energy + 0.05:  # Save only structures within 0.05 eV of the minimum
                        write(f'optimized_wte2_a{a:.3f}_c{c:.3f}.cif', atoms)
                        
                except Exception as e:
                    print(f"Error calculating for a={a}, c={c}: {str(e)}")
                    print("Skipping this configuration and continuing with next...")
                    continue
        
        # Use refined results for final analysis
        results = refined_results

    if opt_a is None or opt_c is None:
        print("No successful calculations completed. Cannot determine optimal values.")
        return results, (None, None)
    
    print(f"Optimized lattice constants: a = {opt_a:.4f} Å, c = {opt_c:.4f} Å")

    # Generate binding curve
    if len(results['c_over_a']) > 0:
        plt.figure(figsize=(10, 6))
        
        # Sort data by c/a ratio for better plotting
        sorted_indices = np.argsort(results['c_over_a'])
        c_over_a_sorted = np.array(results['c_over_a'])[sorted_indices]
        energies_sorted = np.array(results['energies'])[sorted_indices]
        
        # Shift energies so the minimum is at zero
        energies_shifted = energies_sorted - min(energies_sorted)
        
        plt.plot(c_over_a_sorted, energies_shifted, 'o-', linewidth=2, markersize=8)
        plt.xlabel('c/a ratio', fontsize=14)
        plt.ylabel('Relative Energy (eV)', fontsize=14)
        plt.title(r'Binding Curve for 2H-WTe$_2$', fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Add vertical line at optimal c/a ratio
        opt_c_over_a = opt_c / opt_a
        plt.axvline(x=opt_c_over_a, color='r', linestyle='--', 
                    label=f'Optimal c/a = {opt_c_over_a:.4f}')
        
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig('binding_curve.png', dpi=300)
        print(f"Binding curve saved as 'binding_curve.png'")
    else:
        print("No data available to generate binding curve.")
    
    # Report total computation time
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total optimization time: {total_time:.2f} seconds ({total_time/3600:.2f} hours)")
    
    return results, (opt_a, opt_c)


if __name__ == "__main__":
    # Example call (dummy execution):
    trial_wte2 = build_wte2_structure()
    print("\n=== Starting lattice constant optimization ===")
    # Starting with literature values and varying by ±5%
    p_a, p_c = 0.01, 0.01
    a_range = (3.496*(1-p_a), 3.496*(1+p_a), 0.05)  # ±p_a*100% around 3.496 Å
    c_range = (14.07*(1-p_c), 14.07*(1+p_c), 0.2)  # ±p_a*100% around 14.07 Å

    # _ = kpoint_convergence_test(trial_wte2,kmin=4,kmax=4,kstep=1,ecut=400,threshold=1e-4)
    
    # _ = ecut_convergence_test(trial_wte2,(4,4,4),300,350,ecut_step=1,threshold=1e-3)

    # if you don't uncomment the optimal kpoint/cutoff detectors
    optimal_ecut=300
    kpoints = (2,2,1)
    _ = optimize_lattice_constants(a_range=a_range, c_range=c_range, kpts=kpoints, 
                                   ecut=optimal_ecut)