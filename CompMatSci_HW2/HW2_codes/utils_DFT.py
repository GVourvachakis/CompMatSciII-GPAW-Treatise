from ase.build import fcc100, fcc110, fcc111, fcc211
from gpaw import setup_paths
import os
import matplotlib.pyplot as plt
import numpy as np
import pickle, json # for portable and serialized representation of Python objects
from scipy.optimize import curve_fit

# Set up GPAW dataset path
setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
setup_paths[:] = [setup_path]  # Replace all existing paths
os.environ['GPAW_SETUP_PATH'] = setup_path

material: str = "Pd"

def get_parallel_config(kpts):
    """
    Get appropriate parallel configuration based on available cores and k-points.
    Returns a dictionary to pass to GPAW's parallel parameter.
    """
    # Determine if running with MPI
    try:
        from gpaw.mpi import world
        total_cores = world.size
    except ImportError:
        # Fallback if not running with MPI
        total_cores = 1
    
    # Print information about available cores
    print(f"Available cores: {total_cores}")
    
    # If only 1 core available, don't use parallelization
    if total_cores <= 1:
        return {}
    
    # Count number of k-points (if it's a tuple, multiply elements)
    if isinstance(kpts, tuple):
        n_kpts = 1
        for k in kpts:
            n_kpts *= k
    else:
        n_kpts = kpts
    
    # Simple parallelization strategy
    if total_cores <= 4:
        # For few cores, just do domain parallelization
        return {'domain': total_cores}
    else:
        # For more cores, try to balance between domain and k-point parallelization
        # Ensure kpt parallelization doesn't exceed number of k-points
        kpt_cores = min(n_kpts, total_cores // 2)
        
        # Make sure kpt_cores is a divisor of total_cores
        while total_cores % kpt_cores != 0 and kpt_cores > 1:
            kpt_cores -= 1
        
        domain_cores = total_cores // kpt_cores
        
        return {'domain': domain_cores, 'kpt': kpt_cores}

def create_dir(dir_path: str):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def create_slab(surface_type: str, layers: int = 5, vacuum: float = 10.0):
    """Create a slab of specified surface type."""
    if surface_type == '100':
        return fcc100(material, size=(1, 1, layers), vacuum=vacuum)
    elif surface_type == '110':
        return fcc110(material, size=(1, 1, layers), vacuum=vacuum)
    elif surface_type == '111':
        return fcc111(material, size=(1, 1, layers), vacuum=vacuum)
    elif surface_type == '211':
        return fcc211(material, size=(3, 1, layers), vacuum=vacuum) # 3 batches of 5 layers stacked on x-axis
    else:
        raise ValueError(f"Unknown surface type: {surface_type}")

def get_kpoints(surface_type):
    """Get appropriate k-points for the surface type."""
    if surface_type == '211': return (4, 9, 1)
    else: return (9, 9, 1)

def save_results(results, filename=f'surface_energy_results_{material}.pkl'):
    """Save the results to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(results, f)
    print(f"\nResults saved to {filename}")

def load_results(filename=f'surface_energy_results_{material}.pkl'):
    """Load results from a file."""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_layers(results: dict[str,list[float]], surface_type: str | None = None):
    """
    Save selected numerical results to a JSON file.
    """
    with open(f'surface_results/layer_analysis_{material}({surface_type}).json', 'w') as f:
        json.dump({
            'layers': results['layers'],
            'surface_energies_Jm2': results['surface_energies_Jm2'],
            'max_displacements': results['max_displacements'],
            'relaxation_steps': results['relaxation_steps']
        }, f, indent=2)

def apply_constraints(slab, n_layers, n_atoms):
    """
    Fix the bottom layers of the slab by applying a constraint.
    """
    fixed_layers = n_layers // 2  # Fix the bottom half of the layers
    mask = [atom.index < fixed_layers * (n_atoms // n_layers) for atom in slab]
    from ase.constraints import FixAtoms
    slab.set_constraint(FixAtoms(mask=mask))

def ols_surface_energy(results: dict[str,list[float]], surface_type: str | None, bulk_energy_per_atom):
    """
    Perform linear fitting of total energy vs number of layers and calculate
    surface energy using E = N*E_bulk + 2γA
    """

    # Extract data for fitting
    layers = np.array(results['layers'])
    energies = np.array(results['energies'])
    areas = np.array(results['areas'])
    n_atoms = np.array(results['n_atoms'])
    
    # Check if all slabs have approximately the same atoms per layer
    atoms_per_layer = n_atoms / layers
    if not np.allclose(atoms_per_layer, atoms_per_layer[0], rtol=0.01):
        print("Warning: Number of atoms per layer varies between calculations.")
        print(f"Atoms per layer: {atoms_per_layer}")

    # Calculate the average atoms per layer and area
    avg_atoms_per_layer = np.mean(atoms_per_layer)
    avg_area = np.mean(areas)
    
    # Calculate bulk energy per layer (needed for proper layer analysis)
    bulk_energy_per_layer = bulk_energy_per_atom * avg_atoms_per_layer
    

    # Define linear function for fitting: E = m*N_l + b where N_l is number of layers
    # According to E = N_l*E_bulk_per_layer + 2γA
    # m should be close to E_bulk_per_layer and b/(2*A) gives us γ
    linear_func = lambda x, m, b : m * x + b
    
    # Fit energy vs. number of layers
    params, covariance = curve_fit(linear_func, layers, energies)
    fitted_bulk_energy_per_layer, intercept = params
    
    # Calculate surface energy from the intercept
    # Since E = N_l*E_bulk_per_layer + 2γA, we have 2γA = intercept, thus γ = intercept/(2*A)
    fitted_surface_energy = intercept / (2 * avg_area)
    fitted_surface_energy_Jm2 = fitted_surface_energy * 16.022  # Convert to J/m²
    
    # Calculate R-squared for the fit
    residuals = energies - linear_func(layers, *params)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((energies - np.mean(energies))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Create a plot of Energy vs. Number of Layers with the fitted line
    plt.figure(figsize=(10, 6))
    
    # Plot actual data points
    plt.scatter(layers, energies, color='blue', label='DFT Calculations')
    
    # Plot fitted line
    x_fit = np.linspace(min(layers), max(layers), 100)
    y_fit = linear_func(x_fit, *params)
    plt.plot(x_fit, y_fit, 'r-', label=f'Linear Fit: E = {fitted_bulk_energy_per_layer:.4f}*N_l + {intercept:.4f}')
    
    plt.xlabel('Number of Layers (N_l)')
    plt.ylabel('Total Energy (eV)')
    plt.title(f'Energy vs. Number of Layers for {material}({surface_type})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Add text box with the results
    textstr = '\n'.join((
        f'Fitted E_bulk per layer: {fitted_bulk_energy_per_layer:.4f} eV',
        f'Expected E_bulk per layer: {bulk_energy_per_layer:.4f} eV',
        f'Difference: {(fitted_bulk_energy_per_layer-bulk_energy_per_layer):.4f} eV',
        f'Average atoms per layer: {avg_atoms_per_layer:.2f}',
        f'Fitted Surface Energy: {fitted_surface_energy_Jm2:.2f} J/m²',
        f'R²: {r_squared:.4f}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    plt.text(0.05, 0.05, textstr, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='bottom', bbox=props)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(f'{material+surface_type}_energy_vs_layers_fit.png', dpi=300)
    
    # Print the results
    print("\nLinear Fitting Results:")
    print(f"Fitted equation: E = {fitted_bulk_energy_per_layer:.6f}*N_l + {intercept:.6f}")
    print(f"R-squared: {r_squared:.6f}")
    print(f"Average atoms per layer: {avg_atoms_per_layer:.2f}")
    print(f"Fitted bulk energy per layer: {fitted_bulk_energy_per_layer:.6f} eV")
    print(f"Expected bulk energy per layer: {bulk_energy_per_layer:.6f} eV")
    print(f"Difference: {(fitted_bulk_energy_per_layer-bulk_energy_per_layer):.6f} eV")
    print(f"Fitted surface energy (γ): {fitted_surface_energy:.6f} eV/Å² = {fitted_surface_energy_Jm2:.2f} J/m²")
    
    # Add the fitted results to the results dictionary
    results['fitted_bulk_energy_per_layer'] = fitted_bulk_energy_per_layer
    results['expected_bulk_energy_per_layer'] = bulk_energy_per_layer
    results['fitted_intercept'] = intercept
    results['fitted_surface_energy'] = fitted_surface_energy
    results['fitted_surface_energy_Jm2'] = fitted_surface_energy_Jm2
    results['r_squared'] = r_squared
    results['avg_atoms_per_layer'] = avg_atoms_per_layer

def plot_layer_results(results: dict[str,list[float]], surface_type: str | None = None):
    """
    Create and save plots for surface energy versus layers and relaxation metrics.
    """
    create_dir('surface_results/figures')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot surface energy vs layers
    ax1.plot(results['layers'], results['surface_energies_Jm2'], 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Number of Layers', fontsize=12)
    ax1.set_ylabel('Surface Energy (J/m²)', fontsize=12)
    ax1.set_title(f'Surface Energy vs. Number of Layers for {material}({surface_type})', fontsize=14)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Extrapolation to infinite layers if enough data points exist
    if len(results['layers']) >= 3:
        fitting_func = lambda x, a, b : a + b / x # Convergence to 'a' as x -> infinity
        try:
            popt, _ = curve_fit(fitting_func, np.array(results['layers']), np.array(results['surface_energies_Jm2']))
            x_fit = np.linspace(min(results['layers']), max(results['layers']) + 2, 100)
            y_fit = fitting_func(x_fit, *popt)
            ax1.plot(x_fit, y_fit, 'r--', label=f'Extrapolation (γ∞ ≈ {popt[0]:.2f} J/m²)')
            ax1.axhline(y=popt[0], color='r', linestyle=':', alpha=0.5)
            ax1.legend(fontsize=10)
        except Exception as e:
            print("Couldn't perform extrapolation fit:", e)
    
    # Add theoretical surface energy reference if available
    if material == 'Al':
        theoretical_values = {'100': 2.21, '110': 1.95, '111': 1.91, '211': 2.25}
    elif material == 'Pd': 
        theoretical_values = {'100': 2.55, '110': 2.25, '111': 2.21, '211': 2.60} 
    else: raise ValueError(f"Unsupported fcc material: {material}")
    
    if surface_type in theoretical_values:
        theo_value = theoretical_values[surface_type]
        ax1.axhline(y=theo_value, color='g', linestyle='--', label=f'Theoretical: {theo_value} J/m²')
        ax1.legend(fontsize=10)
    
    # Plot relaxation metrics
    ax2.bar(results['layers'], results['max_displacements'], alpha=0.7, label='Max Displacement (Å)')
    ax2.set_xlabel('Number of Layers', fontsize=12)
    ax2.set_ylabel('Maximum Displacement (Å)', fontsize=12, color='C0')
    ax2.tick_params(axis='y', labelcolor='C0')
    
    # Secondary y-axis for relaxation steps
    ax3 = ax2.twinx()
    ax3.plot(results['layers'], results['relaxation_steps'], 'ro-', label='Relaxation Steps')
    ax3.set_ylabel('Relaxation Steps', fontsize=12, color='r')
    ax3.tick_params(axis='y', labelcolor='r')
    
    ax2.set_title('Relaxation Metrics', fontsize=14)
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax3.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    figure_path = f'surface_results/figures/layers_results_{material}({surface_type}).png'
    plt.savefig(figure_path, dpi=300)
    print(f"\nPlot saved to {figure_path}")