from __future__ import print_function
import os
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from ase import Atoms
from ase.parallel import paropen as open
from gpaw import GPAW, PW, FermiDirac, setup_paths, restart
from ase.optimize import QuasiNewton

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

def create_dir(directory):
    os.makedirs(directory, exist_ok=True)

molecule: str = "HCl" # options: "CO", "HCl"

dirpath: str = f"./{molecule}_analysis"

create_dir(dirpath) 

a = 10.  # Size of unit cell (Angstrom)
c = a / 2

# Set experimental bond length based on molecule type
d = 1.22 if molecule=="CO" else 1.27 # Experimental bond length in Å

mol = Atoms(molecule,
            positions=([c - d / 2, c, c],
                        [c + d / 2, c, c]),
            cell=(a, a, a))

calc = GPAW(mode=PW(),
           xc='PBE',
           hund=False,
           eigensolver='rmm-diis',  # This solver can parallelize over bands
           occupations=FermiDirac(0.0, fixmagmom=True),
           txt=f"{dirpath}/{molecule}.out",
           )

mol.calc = calc
e2 = mol.get_potential_energy()
calc.write(f"{dirpath}/{molecule}.gpw")

fd = open(f"{dirpath}/energy_{molecule}.txt", 'w')
print(f" {molecule} molecule energy: %5.2f eV" % e2, file=fd)
fd.close()

mol, calc = restart(f"{dirpath}/{molecule}.gpw", 
                    txt=f"{dirpath}/{molecule}-relaxed.txt")
e2 = mol.get_potential_energy()
d0 = mol.get_distance(0, 1)

fd = open(f"{dirpath}/optimization_{molecule}.txt", 'w')
print('experimental bond length:', file=fd)
print(f'{molecule} molecule energy: %5.2f eV' % e2, file=fd)
print('bondlength : %5.2f Ang' % d0, file=fd)

# Find the theoretical bond length:
relax = QuasiNewton(mol, logfile=f"{dirpath}/qn_{molecule}.log")
relax.run(fmax=0.01) # reduced from 0.05 to improve CO's accuracy
e2 = mol.get_potential_energy()
d0 = mol.get_distance(0, 1)

print(file=fd)
print('PBE energy minimum:', file=fd)
print(f'{molecule} molecule energy: %5.2f eV' % e2, file=fd)
print('bondlength : %5.2f Ang' % d0, file=fd)
fd.close()

# Calculate vibrational frequency by varying bond length around equilibrium
# Save the relaxed geometry and calculator
calc.write(f"{dirpath}/{molecule}_relaxed.gpw")

# Function to calculate energy for different displacements
def energy_vs_displacement(mol, calc, d_eq, displacements) -> tuple[np.ndarray, np.ndarray]:
    """Calculate energy for different displacements around equilibrium bond length."""
    energies = []
    distances = []
    
    c = mol.cell[0][0] / 2  # Center of the box
    
    for disp in displacements:
        # Create new molecule with displaced atoms
        d_new = d_eq + disp
        mol_disp = Atoms(molecule,
                         positions=([c - d_new / 2, c, c],
                                  [c + d_new / 2, c, c]),
                         cell=mol.cell)
        
        mol_disp.calc = calc
        energy = mol_disp.get_potential_energy()
        
        distances.append(d_new)
        energies.append(energy)
        
    return np.array(distances), np.array(energies)

# Define displacements around equilibrium bond length (in Angstrom)
p_disp: float = 0.1
displacements = np.linspace(-p_disp, p_disp, 11)  # 11 points from -p to p (p*100%)

# Calculate energies for different bond lengths
bond_lengths, energies = energy_vs_displacement(mol, calc, d0, displacements)

# Save the data to a file
with open(f"{dirpath}/displacement_energies.txt", 'w') as f:
    f.write('# Bond length (Å)  Energy (eV)\n')
    for d, e in zip(bond_lengths, energies):
        f.write(f'{d:.6f}  {e:.6f}\n')

# Fit a harmonic potential: E = 1/2 * k * x^2 + E0
# Where x is displacement from equilibrium
def harmonic_potential(x, k, E0): return 0.5 * k * x**2 + E0

displacements_from_eq = bond_lengths - d0
fit_params, fit_covariance = curve_fit(harmonic_potential, 
                                       displacements_from_eq, 
                                       energies)

k_force_constant = fit_params[0]  # eV/Å²
E0 = fit_params[1]  # eV

# Convert force constant to vibrational frequency
# Reduced mass based on molecule type (in atomic mass units)
m_C, m_O = (12.011, 15.999)
m_H, m_Cl = (1.008, 35.453) 

m_1, m_2 = (m_C, m_O) if molecule=="CO" else (m_H, m_Cl)

#Reduced masses: CO -> 6.8605 u, HCl -> 0.9801 u
reduced_mass = (m_1 * m_2) / (m_1 + m_2)  # u

# Convert reduced mass to kg
u_to_kg = 1.66053886e-27  # kg/u
reduced_mass_kg = reduced_mass * u_to_kg 

# Convert force constant from eV/Å² to N/m
eV_to_J = 1.602176565e-19  # J/eV
angstrom_to_m = 1e-10  # m/Å
k_SI = k_force_constant * eV_to_J / (angstrom_to_m**2)  # N/m

# Calculate angular frequency (ω = sqrt(k/μ))
omega = np.sqrt(k_SI / reduced_mass_kg)  # rad/s

# Convert to frequency in Hz (f = ω/2π)
freq_Hz = omega / (2 * np.pi)

# Convert to wavenumbers (cm⁻¹) 
# (1 / λ = f / 100*c for [c]=m/s, [f]=1/s, [λ]=cm)
speed_of_light = 299_792_458  # m/s
freq_cm1 = freq_Hz / (speed_of_light * 100)

"""
In spectroscopy, wavenumbers (cm⁻¹) are the standard unit for 
reporting vibrational frequencies because they are directly 
proportional to energy (E = hcν̃, where ν̃ is the wavenumber) and 
allow for easier comparison across different spectroscopic techniques.
"""

# Plot the energy vs. displacement curve
plt.figure(figsize=(10, 6))
plt.scatter(displacements_from_eq, energies, label='DFT Data')

# Plot the fitted harmonic curve
x_fit = np.linspace(min(displacements_from_eq), max(displacements_from_eq), 100)
y_fit = harmonic_potential(x_fit, k_force_constant, E0)
plt.plot(x_fit, y_fit, 'r-', label='Harmonic Fit')

plt.xlabel('Displacement from equilibrium (Å)')
plt.ylabel('Energy (eV)')
plt.title(f'{molecule} Potential Energy Surface')
plt.legend()
plt.grid(True)

plt.savefig(f'{dirpath}/{molecule}_potential_energy.png')
# Experimental vibrational frequency based on molecule type
exp_freq_cm1 = 2143 if molecule=="CO" else 2886 # cm⁻¹

"""
Fundamental Vibrational Frequency (v = 1 ← v = 0): 
--------------------------------------------------
For CO:
-------
2143 cm⁻¹ (Infrared Absorption Spectroscopy) 
[by the Coblentz Society, then compiled in the NIST Chemistry WebBook:
https://cccbdb.nist.gov/exp2x.asp?casno=630080&charge=0&utm_source -> https://link.springer.com/chapter/10.1007/978-1-4757-0961-2_2]


For HCl:
--------
2886 cm⁻¹ (Infrared Absorption Spectroscopy)
[by the Coblentz Society, then compiled in the NIST Chemistry WebBook:
https://webbook.nist.gov/cgi/cbook.cgi?ID=C7647010&Mask=1000&utm_source -> https://cdnsciencepub.com/doi/10.1139/p56-092]
"""

# Print results
fd = open(f'{dirpath}/vibrational_analysis.txt', 'w')
print(f'{molecule} Vibrational Analysis:', file=fd)
print('-----------------------', file=fd)
print(f'Equilibrium bond length: {d0:.4f} Å', file=fd)
print(f'Force constant: {k_force_constant:.4f} eV/Å²', file=fd)
print(f'Force constant: {k_SI:.4e} N/m', file=fd)
print(f'Reduced mass: {reduced_mass:.4f} u', file=fd)
print(f'Vibrational frequency: {freq_Hz:.4e} Hz', file=fd)
print(f'Vibrational frequency: {freq_cm1:.2f} cm⁻¹', file=fd)
print(f'Experimental frequency: {exp_freq_cm1:.2f} cm⁻¹', file=fd)
print(f'Difference: {abs(freq_cm1 - exp_freq_cm1):.2f} cm⁻¹ ({100*abs(freq_cm1 - exp_freq_cm1)/exp_freq_cm1:.2f}%)', file=fd)
fd.close()

print(f"Calculated {molecule} vibrational frequency: {freq_cm1:.2f} cm⁻¹")
print(f"Experimental {molecule} vibrational frequency: {exp_freq_cm1:.2f} cm⁻¹")
print(f"Difference: {abs(freq_cm1 - exp_freq_cm1):.2f} cm⁻¹ ({100*abs(freq_cm1 - exp_freq_cm1)/exp_freq_cm1:.2f}%)")