import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def fermi_dirac(x, sigma):
    """Fermi-Dirac distribution function: f(x,σ) = (1+e^{(x-1)/σ})^{-1}"""
    return 1.0 / (1.0 + np.exp((x - 1.0) / sigma))

def trapezoidal_integration(func, a, b, n):
    """Perform trapezoidal rule integration with n grid points"""
    # Handle case with too few points
    if n < 2:
        return 0
        
    # Generate evenly spaced points
    x = np.linspace(a, b, n)
    dx = (b - a) / (n - 1)
    
    # Evaluate function at grid points
    y = func(x)
    
    # Apply trapezoidal rule
    integral = dx * (0.5 * y[0] + np.sum(y[1:n-1]) + 0.5 * y[n-1])
    
    return integral

def midpoint_integration(func, a, b, n):
    """Perform midpoint rule integration with n intervals"""
    if n < 1:
        return 0
        
    # Generate midpoints of intervals
    dx = (b - a) / n
    x = np.linspace(a + dx/2, b - dx/2, n)
    
    # Evaluate function at midpoints
    y = func(x)
    
    # Apply midpoint rule
    integral = dx * np.sum(y)
    
    return integral

# Define integration range
x_min = 0.0
x_max = 2.0

# Define sigma values to test
sigma_values = [0.01, 0.1, 0.5]

# Calculate "exact" integral using high-precision scipy integration
exact_values = {}
for sigma in sigma_values:
    exact_values[sigma], _ = integrate.quad(lambda x: fermi_dirac(x, sigma), x_min, x_max)
    print(f"Exact integral for σ = {sigma}: {exact_values[sigma]:.10f}")

# Create a sequence of grid points - use fewer points to better see convergence
grid_points = np.unique(np.concatenate([
    np.arange(5, 100, 5),
    np.arange(100, 500, 20),
    np.arange(500, 1001, 50)
]))

# Calculate error convergence using concentrated points near the transition
def analyze_convergence(sigma, method='trapezoidal'):
    errors = []
    integral_values = []
    
    for n in grid_points:
        if method == 'trapezoidal':
            # Define the function for this sigma
            func = lambda x: fermi_dirac(x, sigma)
            approx_value = trapezoidal_integration(func, x_min, x_max, n)
        elif method == 'midpoint':
            func = lambda x: fermi_dirac(x, sigma)
            approx_value = midpoint_integration(func, x_min, x_max, n)
        elif method == 'adaptive':
            # Use adaptive grid with concentration near transition point
            func = lambda x: fermi_dirac(x, sigma)
            # Create non-uniform grid with concentration near x=1
            beta = 3  # Controls concentration near x=1
            if sigma <= 0.01:
                beta = 5  # Higher concentration for smaller sigma
            t = np.linspace(0, 1, n)
            x = x_min + (x_max - x_min) * (t + (1-t)*t**beta)
            dx = np.diff(x)
            x_mid = x[:-1] + dx/2
            y_mid = func(x_mid)
            approx_value = np.sum(y_mid * dx)
        
        error = abs(approx_value - exact_values[sigma])
        errors.append(error)
        integral_values.append(approx_value)
        
    return errors, integral_values

# Create enhanced visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Plot the Fermi-Dirac functions
ax1 = axs[0, 0]
x = np.linspace(0.0, 2.0, 1000)
for sigma in sigma_values:
    ax1.plot(x, fermi_dirac(x, sigma), label=f'σ = {sigma}')
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f', fontsize=12)
ax1.set_title('Fermi-Dirac Distribution', fontsize=14)
ax1.set_xlim(0.0, 2.0)
ax1.set_ylim(0.0, 1.1)
ax1.grid(True)
ax1.legend(fontsize=12)

# Plot convergence of integral values
ax2 = axs[0, 1]
for sigma in sigma_values:
    errors, integral_values = analyze_convergence(sigma)
    ax2.plot(grid_points, integral_values, label=f'σ = {sigma}')
    ax2.axhline(y=exact_values[sigma], color=f'C{sigma_values.index(sigma)}', linestyle='--', alpha=0.5)
ax2.set_xlabel('Number of Grid Points', fontsize=12)
ax2.set_ylabel('Integral Value', fontsize=12)
ax2.set_title('Convergence of Integral Value', fontsize=14)
ax2.grid(True)
ax2.legend(fontsize=12)

# Plot error convergence
ax3 = axs[1, 0]
method = 'trapezoidal'  # Use basic method to show convergence challenges
for sigma in sigma_values:
    errors, _ = analyze_convergence(sigma, method)
    ax3.plot(grid_points, errors, 'o-', label=f'σ = {sigma}', markersize=4)
ax3.axhline(y=0.01, color='r', linestyle='--', label='Target accuracy (0.01)')
ax3.set_xlabel('Number of Grid Points', fontsize=12)
ax3.set_ylabel('Absolute Error', fontsize=12)
ax3.set_title(f'Error Convergence ({method.capitalize()} Method)', fontsize=14)
ax3.set_yscale('log')
ax3.grid(True)
ax3.legend(fontsize=12)

# Create visualization of the adaptive grid
ax4 = axs[1, 1]

# Create demo of grid point distribution for different sigma values
for sigma in sigma_values:
    n_demo = 50  # Use small number to visualize points
    x_demo = np.linspace(0.0, 2.0, n_demo)
    y_demo = np.zeros_like(x_demo)
    ax4.plot(x_demo, y_demo + sigma_values.index(sigma)*0.1, 'o', 
             label=f'Uniform grid, σ = {sigma}', markersize=4)
    
    # Simulate adaptive grid that would be better for small sigma
    beta = 3 if sigma > 0.01 else 5  # Higher concentration for smaller sigma
    t = np.linspace(0, 1, n_demo)
    x_adaptive = x_min + (x_max - x_min) * (t + (1-t)*t**beta)
    y_adaptive = np.zeros_like(x_adaptive)
    ax4.plot(x_adaptive, y_adaptive + 0.05 + sigma_values.index(sigma)*0.1, 'x', 
             label=f'Adaptive grid, σ = {sigma}', markersize=4)
    
    # Also show the function
    x_fine = np.linspace(0.0, 2.0, 1000)
    y_fine = fermi_dirac(x_fine, sigma) * 0.1  # Scale to fit on plot
    ax4.plot(x_fine, y_fine + sigma_values.index(sigma)*0.1, '-', linewidth=1)

ax4.set_xlabel('x', fontsize=12)
ax4.set_title('Grid Point Distribution vs. Function Shape', fontsize=14)
ax4.set_xlim(0.0, 2.0)
ax4.set_yticks([])
ax4.legend(fontsize=10, loc='upper right')

# Add an additional plot to clearly show required grid points for target accuracy
fig2, ax = plt.subplots(figsize=(12, 8))

# Track required grid points for each sigma
required_points = {}

for sigma in sigma_values:
    # Generate more detailed analysis
    detailed_grid = np.arange(5, 1001, 5)
    errors = []
    
    for n in detailed_grid:
        # Use uniform grid integration
        func = lambda x: fermi_dirac(x, sigma)
        approx_value = trapezoidal_integration(func, x_min, x_max, n)
        error = abs(approx_value - exact_values[sigma])
        errors.append(error)
        
        # Determine minimum grid points for target accuracy
        if error < 0.01 and sigma not in required_points:
            required_points[sigma] = n
    
    # Plot error convergence with uniform grid
    ax.plot(detailed_grid, errors, label=f'σ = {sigma}')

# Add target accuracy line
ax.axhline(y=0.01, color='r', linestyle='--', label='Target accuracy (0.01)')

# Add vertical lines for required grid points
for sigma in required_points:
    ax.axvline(x=required_points[sigma], color=f'C{sigma_values.index(sigma)}', 
               linestyle=':', alpha=0.7)
    ax.text(required_points[sigma]+5, 0.02, f'{required_points[sigma]} points',
            color=f'C{sigma_values.index(sigma)}', fontsize=12)

ax.set_xlabel('Number of Grid Points', fontsize=14)
ax.set_ylabel('Absolute Error', fontsize=14)
ax.set_title('Grid Points Required for Target Accuracy (0.01)', fontsize=16)
ax.set_yscale('log')
ax.grid(True)
ax.legend(fontsize=12)

# Print required grid points
print("\nGrid points required for target accuracy (0.01):")
for sigma in required_points:
    print(f"σ = {sigma}: {required_points[sigma]} grid points")

plt.tight_layout()
plt.show()