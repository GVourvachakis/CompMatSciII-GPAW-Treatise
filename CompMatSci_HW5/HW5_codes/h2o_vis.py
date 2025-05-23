import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

"""
Water Molecule Vibrational Modes Visualization

This module provides a visualization of the three fundamental vibrational modes of water (H2O):
1. Bending mode (~1595 cm⁻¹): Also known as the scissors mode, where the H-O-H angle oscillates
2. Symmetric stretching (~3657 cm⁻¹): Both O-H bonds stretch and contract in phase
3. Asymmetric stretching (~3756 cm⁻¹): One O-H bond stretches while the other contracts

The visualization creates a 3D animated representation of these molecular vibrations using
matplotlib, displaying all three modes side by side for comparison. The animation shows how
the atoms move during each vibrational mode, with oxygen atoms in red and hydrogen atoms in blue.

The molecular geometry uses standard values:
- O-H bond length: 0.9575 Å
- H-O-H angle: 104.51°

This visualization is useful for educational purposes in chemistry, spectroscopy, and
molecular physics to help understand the fundamental vibrational modes that contribute
to infrared absorption spectra of water.
"""

def visualize_water_vibrations():
    # Expected frequencies for H2O (in cm-1)
    # These are typical experimental values
    frequencies = {
        0: 1595,  # Bending mode
        1: 3657,  # Symmetric stretching
        2: 3756   # Asymmetric stretching
    }
    
    # Create a figure with 3D plots for each vibrational mode
    fig = plt.figure(figsize=(18, 6))
    
    # Initial water molecule geometry
    d = 0.9575  # O-H bond length in Angstroms
    angle = 104.51  # H-O-H angle in degrees
    t = np.pi / 180 * angle
    
    # Base positions
    O_pos = np.array([0, 0, 0])
    H1_pos = np.array([d, 0, 0])
    H2_pos = np.array([d * np.cos(t), d * np.sin(t), 0])
    
    # Mode 0: Bending mode (scissors)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title(f"Bending Mode (~{frequencies[0]} cm⁻¹)")
    
    # Mode 1: Symmetric stretching
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title(f"Symmetric Stretching (~{frequencies[1]} cm⁻¹)")
    
    # Mode 2: Asymmetric stretching
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title(f"Asymmetric Stretching (~{frequencies[2]} cm⁻¹)")
    
    # Settings for all plots
    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.5, 0.5)
        ax.grid(True)
    
    # Function to animate the molecule
    def update(frame):
        amplitude = 0.2
        freq = 0.1
        
        # Clear previous plots
        ax1.clear()
        ax2.clear()
        ax3.clear()
        
        # Reset labels and limits
        for ax in [ax1, ax2, ax3]:
            ax.set_xlabel('X (Å)')
            ax.set_ylabel('Y (Å)')
            ax.set_zlabel('Z (Å)')
            ax.set_xlim(-1.5, 1.5)
            ax.set_ylim(-1.5, 1.5)
            ax.set_zlim(-0.5, 0.5)
            ax.grid(True)
        
        ax1.set_title(f"Bending Mode (~{frequencies[0]} cm⁻¹)")
        ax2.set_title(f"Symmetric Stretching (~{frequencies[1]} cm⁻¹)")
        ax3.set_title(f"Asymmetric Stretching (~{frequencies[2]} cm⁻¹)")
        
        # Bending vibration - varies the H-O-H angle
        delta_angle = amplitude * np.sin(freq * frame)
        new_t = t + delta_angle
        H1_bend = np.array([d * np.cos(-new_t/2), d * np.sin(-new_t/2), 0])
        H2_bend = np.array([d * np.cos(new_t/2), d * np.sin(new_t/2), 0])
        
        # Symmetric stretching - both O-H bonds stretch in phase
        stretch = amplitude * 0.1 * np.sin(freq * frame)
        H1_sym = np.array([(d + stretch), 0, 0])
        H2_sym = np.array([(d + stretch) * np.cos(t), (d + stretch) * np.sin(t), 0])
        
        # Asymmetric stretching - O-H bonds stretch out of phase
        H1_asym = np.array([(d + stretch), 0, 0])
        H2_asym = np.array([(d - stretch) * np.cos(t), (d - stretch) * np.sin(t), 0])
        
        # Draw molecules
        # Mode 0: Bending
        ax1.scatter(*O_pos, color='red', s=200, label='O')
        ax1.scatter(*H1_bend, color='blue', s=100, label='H')
        ax1.scatter(*H2_bend, color='blue', s=100)
        ax1.plot([O_pos[0], H1_bend[0]], [O_pos[1], H1_bend[1]], [O_pos[2], H1_bend[2]], 'k-')
        ax1.plot([O_pos[0], H2_bend[0]], [O_pos[1], H2_bend[1]], [O_pos[2], H2_bend[2]], 'k-')
        
        # Mode 1: Symmetric stretching
        ax2.scatter(*O_pos, color='red', s=200)
        ax2.scatter(*H1_sym, color='blue', s=100)
        ax2.scatter(*H2_sym, color='blue', s=100)
        ax2.plot([O_pos[0], H1_sym[0]], [O_pos[1], H1_sym[1]], [O_pos[2], H1_sym[2]], 'k-')
        ax2.plot([O_pos[0], H2_sym[0]], [O_pos[1], H2_sym[1]], [O_pos[2], H2_sym[2]], 'k-')
        
        # Mode 2: Asymmetric stretching
        ax3.scatter(*O_pos, color='red', s=200)
        ax3.scatter(*H1_asym, color='blue', s=100)
        ax3.scatter(*H2_asym, color='blue', s=100)
        ax3.plot([O_pos[0], H1_asym[0]], [O_pos[1], H1_asym[1]], [O_pos[2], H1_asym[2]], 'k-')
        ax3.plot([O_pos[0], H2_asym[0]], [O_pos[1], H2_asym[1]], [O_pos[2], H2_asym[2]], 'k-')
        
        ax1.legend(loc='upper right')
        
        return ax1, ax2, ax3
    
    # Create animation
    ani = FuncAnimation(fig, update, frames=100, interval=50, blit=False)
    plt.tight_layout()
    
    return ani, fig

if __name__ == "__main__":
    # Run the visualization
    ani, fig = visualize_water_vibrations()
    plt.show()