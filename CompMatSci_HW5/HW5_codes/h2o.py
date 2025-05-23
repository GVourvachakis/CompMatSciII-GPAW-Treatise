from math import cos, pi, sin
from ase import Atoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from gpaw import GPAW

"""
Vibrational modes of the H2O molecule 
(source: https://gpaw.readthedocs.io/tutorialsexercises/vibrational/vibrations/vibrations.html)
-------------------------------------

DFT can be used to calculate vibrational frequencies of molecules, 
e.g. either in the gas phase or on a surface. 

These results can be compared to experimental output, 
e.g. from IR-spectroscopy, and they can be used to figure out how a molecule 
is bound to the surface. 
In this case we calculate the vibrational frequencies for a water molecule.
"""

import os
from math import pi, cos, sin
from ase import Atoms
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations
from gpaw import GPAW, setup_paths

def setup_gpaw_paths():
    # Set up GPAW dataset path
    setup_path = os.path.expanduser('~/Desktop/DFT_codes/gpaw_datasets/gpaw-setups-0.9.20000')
    setup_paths[:] = [setup_path]  # Replace all existing paths
    os.environ['GPAW_SETUP_PATH'] = setup_path

setup_gpaw_paths()

# Water molecule geometry
d = 0.9575  # O-H bond length in Angstroms
t = pi / 180 * 104.51  # H-O-H angle in radians

# Create water molecule
h2o = Atoms('H2O',
           positions=[(0, 0, 0),  # O atom
                     (d, 0, 0),   # First H atom
                     (d * cos(t), d * sin(t), 0)])  # Second H atom
h2o.center(vacuum=3.5)

# Set up GPAW calculator
h2o.calc = GPAW(txt='h2o.txt',
               mode='lcao',  # Linear combination of atomic orbitals mode
               basis='dzp',   # Double-zeta polarized basis set
               symmetry='off')

# Optimize the structure
print("Optimizing H2O geometry...")
QuasiNewton(h2o).run(fmax=0.05)
print("Geometry optimization complete.")

# Calculate vibrational modes
print("Calculating vibrational modes...")
vib = Vibrations(h2o)
vib.run()

# Summarize results
print("Vibrational analysis summary:")
vib.summary(method='frederiksen')

# Generate visualization files for normal modes
print("Generating mode visualizations...")
for mode in range(9):
    vib.write_mode(mode)
    print(f"Mode {mode} visualization created")

print("Analysis complete. You can visualize the modes with ASE GUI or other molecular viewers.")