from datetime import datetime
from gpaw import setup_paths, GPAW, PW
from ase.build.bulk import bulk
import os
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

# Suppress detailed GPAW output
os.environ['GPAW_VERBOSE'] = '0'
sys.stdout = open(os.devnull, 'w')  # Redirect standard output

# Clear existing paths and set the new one [user-dependent]
intended_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
setup_paths[:] = [intended_path]  # Replace all existing paths
os.environ['GPAW_SETUP_PATH'] = intended_path

sys.stdout = sys.__stdout__  # Restore standard output
print("GPAW looking for datasets in:", setup_paths)
print("Environment GPAW_SETUP_PATH:", os.environ['GPAW_SETUP_PATH'])

# Original code with h=0.2
# h = 0.2
# atoms = bulk("Cu")
# atoms.calc = GPAW(h=0.2, kpts=(8, 8, 8))
# start = datetime.now()
# energy = atoms.get_potential_energy()
# time = datetime.now() - start
# print("Initial Results (h=0.2):", h, energy, time)

"""
1) Generate table of energy and CPU time for different values of h
"""

h_values = [0.30, 0.25, 0.20, 0.18, 0.16, 0.14, 0.12, 0.10]
h_results = []

print("\nTesting different grid spacings (h values)")
print("-" * 60)
print(f"{'h (Å)':^10}{'Energy (eV)':^15}{'Time (s)':^15}{'Relative Time':^15}")
print("-" * 60)

first_time = None
for h in h_values:
    atoms = bulk("Cu")
    atoms.calc = GPAW(h=h, kpts=(8, 8, 8), maxiter=200, 
                     convergence={'energy': 1e-6, 'density': 1e-6})
    start = datetime.now()
    energy = atoms.get_potential_energy()
    elapsed_time = datetime.now() - start
    time_seconds = elapsed_time.total_seconds()
    
    if first_time is None:
        first_time = time_seconds
    
    relative_time = time_seconds / first_time
    h_results.append([h, energy, time_seconds, relative_time])
    
    print(f"{h:^10.2f}{energy:^15.4f}{time_seconds:^15.2f}{relative_time:^15.2f}")

# Save grid spacing results to CSV
with open("grid_spacing_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Grid Spacing (h)", "Energy (eV)", "Time (s)", "Relative Time"])
    writer.writerows(h_results)

# Original PW code
# my_pw = 300
# atoms = bulk("Cu")
# atoms.calc = GPAW(mode=PW(my_pw), kpts=(8, 8, 8))
# start = datetime.now()
# energy = atoms.get_potential_energy()
# time = datetime.now() - start
# print(f"\nInitial Results (PW={my_pw} eV):", my_pw, energy, time)

"""
2) Grid mode amd Plane-wave mode execution time comparison
"""

pw_values = [100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]
pw_results = []

print("\nTesting different plane-wave cutoffs")
print("-" * 60)
print(f"{'PW Cutoff (eV)':^15}{'Energy (eV)':^15}{'Time (s)':^15}{'Relative Time':^15}")
print("-" * 60)

first_pw_time = None
for pw in pw_values:
    atoms = bulk("Cu")
    atoms.calc = GPAW(mode=PW(pw), kpts=(8, 8, 8), maxiter=200,
                     convergence={'energy': 1e-6, 'density': 1e-6})
    start = datetime.now()
    energy = atoms.get_potential_energy()
    elapsed_time = datetime.now() - start
    time_seconds = elapsed_time.total_seconds()
    
    if first_pw_time is None:
        first_pw_time = time_seconds
    
    relative_time = time_seconds / first_pw_time
    pw_results.append([pw, energy, time_seconds, relative_time])
    
    print(f"{pw:^15}{energy:^15.4f}{time_seconds:^15.2f}{relative_time:^15.2f}")

# Save PW results to CSV
with open("plane_wave_results.csv", mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["PW Cutoff (eV)", "Energy (eV)", "Time (s)", "Relative Time"])
    writer.writerows(pw_results)

# Calculate average time for grid vs PW
avg_grid_time = sum(row[2] for row in h_results) / len(h_results)
avg_pw_time = sum(row[2] for row in pw_results) / len(pw_results)

print("\n--Grid vs. Plane-wave mode comparison--")
print(f"Average time for grid mode: {avg_grid_time:.2f} seconds")
print(f"Average time for plane-wave mode: {avg_pw_time:.2f} seconds")
if avg_grid_time < avg_pw_time:
    print("Grid mode is faster than plane-wave mode on average.")
else:
    print("Plane-wave mode is faster than grid mode on average.")


"""
3) Plot of potential energy of fcc Cu as a function of cutoff energy.
"""
# E_tot vs. E_cutoff plot
plt.figure(figsize=(10, 6))
pw_values_plot = [row[0] for row in pw_results]
pw_energies = [row[1] for row in pw_results]

plt.plot(pw_values_plot, pw_energies, 'o-', linewidth=2, markersize=8)
plt.xlabel('Plane-Wave Cutoff Energy (eV)', fontsize=12)
plt.ylabel('Total Energy (eV)', fontsize=12)
plt.title('Energy Convergence of fcc Cu with Plane-Wave Cutoff', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
# plt.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()

# Save the plot
plt.savefig('cu_pw_convergence.png', dpi=300)
print("\nPlot saved as 'cu_pw_convergence.png'")

# Compare convergence by finding when energy difference becomes < 0.01 eV
converged_pw = None
for i in range(1, len(pw_results)):
    if abs(pw_results[i][1] - pw_results[i-1][1]) < 0.01:
        converged_pw = pw_results[i][0]
        break

converged_h = None
for i in range(1, len(h_results)):
    if abs(h_results[i][1] - h_results[i-1][1]) < 0.01:
        converged_h = h_results[i][0]
        break

print("\nConvergence Summary:")
if converged_pw:
    print(f"Plane-wave energy converged to < 0.01 eV at cutoff: {converged_pw} eV")
else:
    print("Plane-wave energy did not converge to < 0.01 eV with tested values")

if converged_h:
    print(f"Grid mode energy converged to < 0.01 eV at h: {converged_h} Å")
else:
    print("Grid mode energy did not converge to < 0.01 eV with tested values")