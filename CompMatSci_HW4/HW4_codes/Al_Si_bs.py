#!/usr/bin/env python3
"""
Band structure calculation for Al (FCC) and Si (diamond).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk
from gpaw import setup_paths, GPAW, PW, FermiDirac

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

# Global variable to define the crystalline material
material: str = "Si"  # Options: "Al" or "Si"

def calculate_band(use_soc=False):
    """
    Calculate and plot band structure for the given material.
    
    Parameters:
    -----------
    material : str
        Chemical symbol for the material ('Al' or 'Si')
    use_soc : bool
        Enable spin-orbit coupling calculation
    """
    if use_soc: # non-collinear spin calculation mode
        xc='LDA' # Mostly used in the literature for SOC. We should keep PBE for comparison.
        atom_per_cell = 1 if material=="Al" else 2
        experimental={'magmoms': np.zeros((atom_per_cell, 3)), 'soc': True}
        # 2: The number of atoms in the primitive cell  
        # (Si has 2 atoms in its primitive cell while Al has 1)
        # 3: The three spatial dimensions (x, y, z) for each magnetic moment vector
        
        # For SOC calculations, use fewer bands
        nbands_value = 8  # or even higher if necessary (16 for Si)
        conv_bands = 4  # (8 for Si)
    else:
        xc, experimental='PBE', {}
        # For non-SOC calculations, use more bands
        nbands_value = 16
        conv_bands = 8

    # Set up the calculation parameters
    if material == 'Al':
        # Al has FCC structure
        structure: str = 'fcc'
        a: float = 4.05
        emin, emax = -10, 10  # Energy range for plot
        title = 'Aluminum (FCC)'
        # Al is a metal, no band gap
    elif material == 'Si':
        # Si has diamond structure
        structure: str = 'diamond'
        a: float = 5.43
        emin, emax = -12, 8  # Energy range for plot
        title = 'Silicon (Diamond)'
    else:
        raise ValueError(f"Unsupported material: {material}")
    
    kpts = 8  # k-point density
    band_path = 'GXWKGLUWLK'
    band_points = 60
    atoms = bulk(material, structure, a)

    # Standard ground state calculation (with plane wave basis)
    # Ground state calculation
    calc = GPAW(
                    mode=PW(400),
                    kpts={'size': (kpts, kpts, kpts), 'gamma': True},
                    xc=xc,
                    #random=True,  # random guess (needed if many empty bands required)
                    experimental=experimental,
                    symmetry='off',  # Turn off symmetry if you want full BZ
                    occupations=FermiDirac(0.01),  # Small broadening for better convergence
                    txt=f'{material.lower()}_gs.txt'
                )
    atoms.calc = calc
    atoms.get_potential_energy() # run calculation
    # Get the Fermi level
    ef = calc.get_fermi_level()
    print(f"Ground state Fermi level: {ef:.4f} eV")
    # Save the ground state calculation
    calc.write(f'{material.lower()}_gs.gpw')
    
    # Calculate band structure along high symmetry path
    bp = atoms.cell.bandpath(band_path, npoints=band_points)
    bp.plot()

    # Restart from ground state and fix density: (or atoms.calc.fixed_density(...) )
    # Notice: using a localized basis ('dzp') may need to be reconsidered for SOC.
    bs_calc = GPAW(f'{material.lower()}_gs.gpw').fixed_density(
                                        nbands=nbands_value, # high value for accurate unoccupied bands
                                        basis='dzp',
                                        symmetry='off',
                                        kpts=bp,
                                        convergence={'bands':conv_bands})  # Adjust convergence bands for SOC

    bs = bs_calc.band_structure()
    ef = bs_calc.get_fermi_level()
    print(f"Fermi level: {ef:.4f} eV")

    bs = bs.subtract_reference() # ground on Fermi level
    
    soc_label = "-soc" if use_soc else ""
    
    # Save the band structure
    bs.write(f"{material.lower()}_bs.json")

    # Plot the band structure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    bs.plot(ax=ax, emin=emin, emax=emax)  # eref not needed since we used subtract_reference()
    with_soc: str = f" ({soc_label[1:]})" if use_soc else ""
    plt.title(f"Band Structure of {title+with_soc}")
    plt.ylabel(r"Energy - E$_F$ (eV)")
    plt.tight_layout()

    plt.savefig(f'./{material.lower()}{soc_label}_bs.png', dpi=300)
    # equivalent plot:
    # bs.plot(filename=f'{material.lower()}{soc_label}-bs.png', show=True, emin=emin, emax=emax)
    plt.show()

    if use_soc and material=="Si": # zoom-in for SOC to reproduce figure in GPAW tutorial
        bs.plot(filename=f'zoomed_{material.lower()}{soc_label}_bs.png', 
                show=True, emin=-1.0, emax=0.5)

    print(f"Band structure calculation for {material}{soc_label} completed.")
    return bs

if __name__ == "__main__":
    # Calculate band structure for Al (FCC) or Si (Diamond)
    print(f"Calculating band structure for {material} without SOC...")
    bs_run = calculate_band(use_soc=False)

    print(f"Calculating band structure for {material} with SOC...")
    bs_run = calculate_band(use_soc=True)