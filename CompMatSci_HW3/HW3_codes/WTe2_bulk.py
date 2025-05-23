import numpy as np
import matplotlib.pyplot as plt
from ase.io import write
from ase.eos import EquationOfState
from ase.constraints import FixSymmetry
from gpaw import GPAW, PW
# from dftd3.ase import DFTD3
# from ase.calculators.mixing import SumCalculator

from WTe2_search import setup_gpaw_paths

setup_gpaw_paths()

# Step 5: Calculate bulk modulus using volume scaling
def calculate_bulk_modulus(atoms, kpts, ecut, scale_range=0.94, scale_steps=9, isothermal="murnaghan"):
    """
    Calculate the bulk modulus by fitting energy vs volume curve.
    
    Args:
        atoms: ASE Atoms object with optimized structure
        kpts: K-point grid to use
        ecut: Plane wave cutoff energy in eV
        scale_range: Minimum scale factor (e.g., 0.94 for 6% compression)
        scale_steps: Number of volume points to sample
        isothermal: Equation of state for B(V)
        
    Returns:
        Bulk modulus in GPa, equilibrium volume, and minimum energy
    """
    # Generate scale factors symmetrically distributed around 1.0
    scale_factors = np.linspace(scale_range, 1/scale_range, scale_steps)
    
    volumes = []
    energies = []
    
    # Add symmetry constraint to preserve hexagonal structure
    atoms_copy = atoms.copy()
    atoms_copy.set_constraint(FixSymmetry(atoms_copy))
    
    for _, scale in enumerate(scale_factors):
        scaled_atoms = atoms_copy.copy()
        scaled_atoms.set_cell(atoms_copy.get_cell() * scale, scale_atoms=True)
        
        volume = scaled_atoms.get_volume()
        
        # Create GPAW calculator with vdW corrections
        base_calc = GPAW(mode=PW(ecut),
                    kpts=kpts,
                    txt=f'eos_scale_{scale:.3f}.txt',
                    xc='PBE',
                    parallel={'sl_auto': True})
        
        # Add DFT-D3 van der Waals correction
        # calc = SumCalculator([base_calc, DFTD3(method='PBE')])
        scaled_atoms.calc = base_calc
        energy = scaled_atoms.get_potential_energy()
        
        volumes.append(volume)
        energies.append(energy)
        print(f"Scale: {scale:.3f}, Volume: {volume:.3f} Å³, Energy: {energy:.6f} eV")
        
        # Save the structure for verification
        write(f'wte2_scale_{scale:.3f}.cif', scaled_atoms)
    
    # Fit equation of state - Murnaghan EOS
    eos = EquationOfState(volumes, energies, eos=isothermal)
    v0, e0, B = eos.fit()
    
    B_GPa = B * 160.21766
    
    print(f"Equation of State Results:")
    print(f"Equilibrium volume: {v0:.3f} Å³")
    print(f"Minimum energy: {e0:.6f} eV")
    print(f"Bulk modulus: {B_GPa:.2f} GPa")
    
    # Plot EOS
    plt.figure(figsize=(10, 6))
    eos.plot(filename=f"eos_wte2_{isothermal}.png")
    
    return B_GPa, v0, e0