from datetime import datetime
from gpaw import setup_paths, GPAW
from ase.build.bulk import bulk
import os
import csv
import sys

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

# Prepare to store results
results = []
first_time = None
atoms = bulk("Cu")


for M in range(1, 11):
    atoms.calc = GPAW(mode='pw', h=0.1, kpts=(M, M, M), maxiter=200, 
                     convergence={'energy': 1e-6, 'density': 1e-6}) #, symmetry="off")
    start = datetime.now()
    energy = atoms.get_potential_energy()
    elapsed_time = datetime.now() - start
    
    # Get number of k-points in the IBZ
    n_ibz = len(atoms.calc.get_ibz_k_points())
    
    # Store first time for relative time calculation
    if M == 1:
        first_time = elapsed_time.total_seconds()
    
    # Calculate relative time (τₘ/τ₁)
    relative_time = elapsed_time.total_seconds() / first_time
    
    # Store energy per atom (Cu has 1 atom in the unit cell)
    energy_per_atom = energy
    
    # Round values to match the format in the table
    energy_per_atom_rounded = round(energy_per_atom, 4)
    relative_time_rounded = round(relative_time, 1)
    
    results.append([M, energy_per_atom_rounded, n_ibz, relative_time_rounded])
    
    print(f"M={M}: E/atom={energy_per_atom_rounded} eV, IBZ k-points={n_ibz}, Time ratio={relative_time_rounded}")

def main() -> None:

    # Save results to a CSV file
    csv_filename = "gpaw_experiment_results.csv"
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["M", "E/atom (eV)", "No. of k Points in IBZ", "τₘ/τ₁"])
        writer.writerows(results)
    print(f"Results saved to {csv_filename}")
    
    # Print the table in the format shown in the image
    print("\nTABLE 3.2  Results from Computing the Total Energy of fcc Cu with")
    print("M × M × M k Points Generated Using the Monkhorst-Pack Method")
    print("-" * 60)
    print(f"{'M':^5}{'E/atom (eV)':^15}{'No. of k Points in IBZ':^25}{'τₘ/τ₁':^10}")
    print("-" * 60)
    
    for row in results:
        print(f"{row[0]:^5}{row[1]:^15}{row[2]:^25}{row[3]:^10}")

if __name__ == "__main__": main()