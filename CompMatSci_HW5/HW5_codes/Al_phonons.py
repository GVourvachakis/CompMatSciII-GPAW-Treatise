# TO DO: Follow the slide before the last one on phonons course.

"""
Phonon calculations
-------------------
Module for calculating vibrational normal modes for periodic 
systems using the so-called small displacement method 
(see e.g. [Alfe]). 

So far, space-group symmetries are not exploited to reduce 
the number of atomic displacements that must be calculated 
and subsequent symmetrization of the force constants.

For polar materials the dynamical matrix at the zone 
center acquires a non-analytical contribution 
that accounts for the LO-TO splitting. 
This contribution requires additional functionality 
to evaluate and is not included in the present implementation. 

Its implementation in conjunction with the small 
displacement method is described in [Wang].

Calculating the phonon dispersion for bulk aluminum using 
a 7x7x7 supercell within effective medium theory.
"""

# Plot the band structure and DOS:
import matplotlib.pyplot as plt  # noqa

from ase.build import bulk
from ase.calculators.emt import EMT
from ase.phonons import Phonons

# Setup crystal and EMT calculator
atoms = bulk('Al', 'fcc', a=4.05)

# Phonon calculator
N = 7
ph = Phonons(atoms, EMT(), supercell=(N, N, N), delta=0.05)
ph.run()

# Read forces and assemble the dynamical matrix
ph.read(acoustic=True)
ph.clean()

path = atoms.cell.bandpath('GXULGK', npoints=100)
bs = ph.get_band_structure(path)

dos = ph.get_dos(kpts=(20, 20, 20)).sample_grid(npts=100, width=1e-3)

fig = plt.figure(figsize=(7, 4))
ax = fig.add_axes([.12, .07, .67, .85])

emax = 0.035
bs.plot(ax=ax, emin=0.0, emax=emax)

dosax = fig.add_axes([.8, .07, .17, .85])
dosax.fill_between(dos.get_weights(), dos.get_energies(), y2=0, color='grey',
                   edgecolor='k', lw=1)

dosax.set_ylim(0, emax)
dosax.set_yticks([])
dosax.set_xticks([])
dosax.set_xlabel("DOS", fontsize=18)

fig.savefig('Al_phonon.png')

"""
WARNING, 2 imaginary frequencies at q = ( 0.00,  0.00,  0.00) ; (omega_q = 8.501e-09*i)
WARNING, 2 imaginary frequencies at q = ( 0.00,  0.00,  0.00) ; (omega_q = 8.501e-09*i)
"""

# More Inspection:
# ----------------
# from ase.io.trajectory import Trajectory
# from ase.io import write

# # Write modes for specific q-vector to trajectory files:
# L = path.special_points['L']
# ph.write_modes([l / 2 for l in L], branches=[2], repeat=(8, 8, 8), kT=3e-4,
#                center=True)


# # Generate gif animation:
# # XXX Temporarily disabled due to matplotlib writer compatibility issue.
# with Trajectory('phonon.mode.2.traj', 'r') as traj:
#     write('Al_mode.gif', traj, interval=50,
#           rotation='-36x,26.5y,-25z')
