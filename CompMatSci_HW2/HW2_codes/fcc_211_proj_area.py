import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

material : str = "Al"

if material == "Al": a=4.05 # Experimentally measured via XRD in Å
elif material == "Pd": a=3.89 # Experimentally measured via XRD in Å 
else: raise ValueError(f"Unsupported fcc material: {material}")


# Compute the projected area for (211):
# A_proj = (a^2 * sqrt(6))/2
A_proj = (a**2 * np.sqrt(6)) / 2

# Define two in-plane lattice vectors that span the projected unit cell.
# One convenient choice is:
#   u = (a, 0)
#   v = (a/2, a*sqrt(6)/2)
# Their cross product has magnitude: |u x v| = a * (a*sqrt(6)/2) = (a^2*sqrt(6))/2 = A_proj.
u = np.array([a, 0])
v = np.array([a/2, a*np.sqrt(6)/2])

# Coordinates for the projected parallelogram vertices:
P0 = np.array([0, 0])
P1 = u
P2 = u + v
P3 = v
parallelogram = np.array([P0, P1, P2, P3])

# Create a figure to show both the stepped contour and the projected area.
fig, ax = plt.subplots(figsize=(8, 8))

# -----------------------------
# Plot a schematic of the actual (211) stepped surface.
# This is a simplified representation of the atomic contour.
# (The coordinates below are illustrative.)
stepped_points = np.array([
    [0, 0],
    [a, 0],
    [a, 0.4*a],
    [0.8*a, 0.4*a],
    [0.8*a, 0.8*a],
    [0, 0.8*a]
])
stepped_poly = patches.Polygon(stepped_points, closed=True, edgecolor='red', 
                                 facecolor='none', linestyle='--', lw=1.5, label='Stepped Contour')
ax.add_patch(stepped_poly)
for pt in stepped_points:
    ax.plot(pt[0], pt[1], 'ro', markersize=4)

# -----------------------------
# Plot the projected unit cell (parallelogram) in blue.
proj_poly = patches.Polygon(parallelogram, closed=True, edgecolor='blue', 
                            facecolor='none', lw=2, label='Projected Unit Cell')
ax.add_patch(proj_poly)
for pt in parallelogram:
    ax.plot(pt[0], pt[1], 'bo', markersize=6)

# Draw the lattice vectors u and v as arrows from the origin.
ax.arrow(0, 0, u[0], u[1], head_width=0.2*a, head_length=0.2*a, fc='green', ec='green', lw=2)
ax.text(u[0]/2, u[1]-0.05*a, r'$\mathbf{u}$', color='green', fontsize=14, ha='center')
ax.arrow(0, 0, v[0], v[1], head_width=0.2*a, head_length=0.2*a, fc='purple', ec='purple', lw=2)
ax.text(v[0]/2, v[1]/2, r'$\mathbf{v}$', color='purple', fontsize=14, ha='right')

# Annotate the area of the projected cell.
ax.text(np.mean(parallelogram[:,0]), np.mean(parallelogram[:,1]) + 0.3*a,
        r'$A_{proj}^{(211)}=\frac{a^2\sqrt{6}}{2}\approx$' + f'\n {A_proj:.2f} Å$^2$', 
        fontsize=14, color='blue', ha='center', bbox=dict(facecolor='white', alpha=0.7, edgecolor='blue'))

# Add labels and title.
ax.set_title('FCC (211) Facet: Stepped Contour vs. Projected Unit Cell', fontsize=16)
ax.set_xlabel('x (Å)', fontsize=14)
ax.set_ylabel('y (Å)', fontsize=14)
ax.legend(loc='upper right', fontsize=12)

# Set limits to provide some margins.
ax.set_xlim(-0.5*a, 1.5*a)
ax.set_ylim(-0.5*a, 1.5*a)
ax.set_aspect('equal')
ax.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig("detailed_fcc_211_projected_area.png", dpi=150)
plt.show()