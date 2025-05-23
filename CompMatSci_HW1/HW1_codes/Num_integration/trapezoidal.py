import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy import integrate

def f(x : np.ndarray) -> np.ndarray:
    """The function to integrate: f(x) = x^2"""
    return x**2

def trapezoidal_rule(f: np.ndarray, a: float, b: float, n: int) -> float:
    """
    Perform trapezoidal rule integration on function f from a to b using n subintervals
    
    Parameters:
    -----------
    f : function
        The function to integrate
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    n : int
        Number of partitions (npoints - 1)
        
    Returns:
    --------
    float
        Approximate value of the integral
    """
    # Handle case when n=1 (npoints=2)
    if n == 1:
        return (b - a) * (f(a) + f(b)) / 2
    
    # Generate evenly spaced points
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Apply trapezoidal rule formula
    integral = (b - a) / (2 * n) * (y[0] + y[-1] + 2 * np.sum(y[1:-1]))
    
    return integral

def exact_integral(f, a, b):
    """Calculate the exact integral using scipy.integrate.quad"""
    result, _ = integrate.quad(f, a, b)
    return result

def visualize_trapezoidal(f: np.ndarray, a: float, b: float, n: int, 
                          ax: matplotlib.axes.Axes | None = None) -> None:
    """
    Visualize the trapezoidal rule approximation for a function
    
    Parameters:
    -----------
    f : function
        The function to integrate
    a : float
        Lower limit of integration
    b : float
        Upper limit of integration
    n : int
        Number of subintervals (npoints - 1)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 6))
    
    # Generate points for plotting the actual function
    x_fine = np.linspace(a, b, 1000)
    y_fine = f(x_fine)
    
    # Generate evenly spaced points for trapezoidal rule
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    # Plot the actual function
    ax.plot(x_fine, y_fine, 'b-', linewidth=2, label='f(x) = x²')
    
    # Plot the trapezoidal approximation
    for i in range(n):
        # Plot the trapezoid
        trapezoid_x = [x[i], x[i+1], x[i+1], x[i]]
        trapezoid_y = [0, 0, y[i+1], y[i]]
        ax.fill(trapezoid_x, trapezoid_y, 'r', alpha=0.2)
        
        # Plot the linear approximation segment
        ax.plot([x[i], x[i+1]], [y[i], y[i+1]], 'r-', linewidth=2)
    
    # Plot points
    ax.plot(x, y, 'ro', markersize=6)
    
    # Mark the endpoints
    ax.plot([a, a], [0, f(a)], 'k--', linewidth=1)
    ax.plot([b, b], [0, f(b)], 'k--', linewidth=1)
    ax.text(a, -0.1, 'a', ha='center', va='top', fontsize=12)
    ax.text(b, -0.1, 'b', ha='center', va='top', fontsize=12)
    
    # Set labels and title
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)
    ax.set_title(f'Trapezoidal Rule for f(x) = x² with {n+1} points', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Set limits with some padding
    ax.set_xlim(a - 0.1, b + 0.1)
    ax.set_ylim(0, f(b) + 0.5)

# Set integration limits
a = 1.0
b = 2.0

# Calculate exact value
exact_value = exact_integral(f, a, b)
print(f"Exact integral of f(x) = x² from {a} to {b}: {exact_value}")

# Create figure for visualization
fig, ax = plt.subplots(figsize=(10, 6))

# Visualize with npoints = 2 (n = 1)
visualize_trapezoidal(f, a, b, 1, ax)

# Calculate and store errors for different numbers of points
results = []
for npoints in range(2, 11):  # from 2 to 10 points (1 to 9 subintervals)
    n = npoints - 1  # Convert to number of subintervals
    approx_value = trapezoidal_rule(f, a, b, n)
    error = abs(approx_value - exact_value)
    relative_error = error / exact_value * 100
    results.append([npoints, n, approx_value, error, relative_error])

# Display results in a table
headers = ["npoints", "subintervals", "Approximation", "Absolute Error", "Relative Error (%)"]
print("\n" + tabulate(results, headers=headers, floatfmt=".10f"))

# Create a figure for the error convergence
fig2, ax2 = plt.subplots(figsize=(10, 6))
npoints_range = np.arange(2, 11)
errors = [row[3] for row in results]

ax2.plot(npoints_range, errors, 'bo-', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Points (npoints)', fontsize=12)
ax2.set_ylabel('Absolute Error', fontsize=12)
ax2.set_title('Error Convergence of Trapezoidal Rule', fontsize=14)
ax2.grid(True)
ax2.set_yscale('log')

plt.tight_layout()
plt.show()