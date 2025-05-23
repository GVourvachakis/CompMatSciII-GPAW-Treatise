import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List

def fixed_point_iteration(f: np.ndarray, x0: float, n_iter: int = 10) -> List[float]:
    """
    Perform fixed point iteration for a function f starting from x0.
    
    Parameters:
    -----------
    f : function
        The function to iterate
    x0 : float
        Initial value
    n_iter : int
        Number of iterations
        
    Returns:
    --------
    list
        Sequence of iterations
    """
    x = x0
    sequence = [x]
    
    for _ in range(n_iter):
        try:
            x = f(x)
            sequence.append(x)
            
            # Check for convergence or divergence
            if not np.isfinite(x) or abs(x) > 1e6:
                break
        except:
            break
            
    return sequence

def plot_fixed_point_iteration(f: np.ndarray, x0: float, n_iter: int = 10, ax: matplotlib.axes.Axes | None = None, 
                               title : str = "", color: str = 'blue') -> None:
    """
    Plot the fixed point iteration for a function f.
    
    Parameters:
    -----------
    f : function
        The function to iterate
    x0 : float
        Initial value
    n_iter : int
        Number of iterations
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    title : str, optional
        Title for the plot
    color : str, optional
        Color for the plot
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    # Get the sequence of iterations
    sequence = fixed_point_iteration(f, x0, n_iter)
    
    # Plot f(x)
    x = np.linspace(-1.5, 1.5, 1_000)
    y = np.array([f(xi) if np.isfinite(f(xi)) and abs(f(xi)) < 10 else np.nan for xi in x])
    ax.plot(x, y, color=color)
    
    # Plot y = x line
    ax.plot(x, x, 'k--', alpha=0.3)
    
    # Plot the iterations
    for i in range(1, len(sequence)):
        # Draw horizontal line from (x_{n-1}, x_n) to (x_n, x_n)
        ax.plot([sequence[i-1], sequence[i]], [sequence[i], sequence[i]], color='brown', linewidth=1)
        # Draw vertical line from (x_n, x_n) to (x_n, x_{n+1})
        if i < len(sequence) - 1:
            ax.plot([sequence[i], sequence[i]], [sequence[i], sequence[i+1]], color='brown', linewidth=1)
    
    # Plot the sequence points
    ax.scatter(sequence[:-1], sequence[1:], color='red', s=30, zorder=5)
    
    # Add zoomed region similar to the uploaded image if sequence converges
    if len(sequence) > 5 and abs(sequence[-1] - sequence[-2]) < 0.01:
        # Create a rectangle patch around the convergence point
        center_x = sequence[-1]
        center_y = sequence[-1]
        rect_size = 0.2
        
        # Add rectangle
        rect = Rectangle((center_x - rect_size/2, center_y - rect_size/2), 
                         rect_size, rect_size, 
                         linewidth=1, edgecolor='brown', facecolor='none')
        ax.add_patch(rect)
        
        # Add a zoom inset
        inset_ax = ax.inset_axes([0.6, 0.6, 0.3, 0.3])
        inset_ax.plot(x, y, color=color)
        inset_ax.plot(x, x, 'k--', alpha=0.3)
        inset_ax.scatter(sequence[-5:], [f(seq_val) for seq_val in sequence[-5:]], color='red', s=30, zorder=5)
        
        # Connect the last iterations with lines
        for i in range(max(len(sequence)-5, 1), len(sequence)):
            # Draw horizontal line
            inset_ax.plot([sequence[i-1], sequence[i]], [sequence[i], sequence[i]], color='brown', linewidth=1)
            # Draw vertical line
            if i < len(sequence) - 1:
                inset_ax.plot([sequence[i], sequence[i]], [sequence[i], sequence[i+1]], color='brown', linewidth=1)
        
        # Set limits for the inset
        inset_ax.set_xlim(center_x - rect_size/2, center_x + rect_size/2)
        inset_ax.set_ylim(center_y - rect_size/2, center_y + rect_size/2)
        inset_ax.set_xticks([])
        inset_ax.set_yticks([])
        inset_ax.spines['top'].set_color('brown')
        inset_ax.spines['bottom'].set_color('brown')
        inset_ax.spines['left'].set_color('brown')
        inset_ax.spines['right'].set_color('brown')
    
    # Set the axis limits
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.2, 1.2)
    
    # Set labels and title
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

# Define the functions
def f1(x):
    return np.cos(x)

def f2(x):
    return np.cos(x**2 * np.sin(x))

def f3(x):
    return np.sqrt(abs(x)) - np.sin(x)  # Using abs to handle negative values

def f4(x):
    return 3.25 * x - 1.5 * x**2

# Create the figure with 2x2 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

# Initial value
x0 = 1


def main() -> None:
    # Plot each function
    plot_fixed_point_iteration(f1, x0, n_iter=10, ax=axs[0], 
                            title=r"$f(x) = \cos(x)$", color='blue')
    plot_fixed_point_iteration(f2, x0, n_iter=10, ax=axs[1], 
                            title=r"$f(x) = \cos(x^2 \sin(x))$", color='blue')
    plot_fixed_point_iteration(f3, x0, n_iter=10, ax=axs[2], 
                            title=r"$f(x) = \sqrt{x} - \sin(x)$", color='blue')
    plot_fixed_point_iteration(f4, x0, n_iter=10, ax=axs[3], 
                            title=r"$f(x) = 3.25x - 1.5x^2$ (Logistic Map)", color='blue')

    # Adjust layout
    plt.tight_layout()
    plt.show()

if __name__ == "__main__": main()