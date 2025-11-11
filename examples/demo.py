"""
Example script demonstrating SSH model and Berry phase calculations.

This script generates four figures for the results section:
1. Energy bands
2. h(k) ellipse in parameter space
3. Closed loop in (Delta, delta) space
4. Berry phases vs. point index
"""

import sys
import os

# Add the project root directory to the Python path
# This allows the script to be run directly or as a module
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.model import BerryBandModel
from src.loops import create_closed_loop_circle, calculate_berry_phases_for_parameters
from src.plots import plot_energy_bands, plot_closed_loop, plot_berry_phases, plot_h_ellipse, plot_h_ellipse_3d, plot_h_trajectories_3d, plot_key_points_h_trajectories_3d
import matplotlib.pyplot as plt

# ============================================================================
# Model parameters - modify these values as needed
# ============================================================================
# Initial values for Delta and delta (for energy bands and h(k) ellipse plots)
INITIAL_DELTA = -0.25  #Site energy alternation
INITIAL_DELTA_SMALL = 0 # Hopping alternation

# Model parameters
E_P = -6.0  # Average site energy
T = -2.8  # Average hopping
A = 1.0  # Lattice constant
ORBITAL_TYPE = 'midbond'  # 'midbond' or 'site'

# Closed loop parameters (for Berry phase calculation)
LOOP_CENTER = (0, 0)  # (Delta, delta) center of the loop
LOOP_RADIUS = 0.5  # Radius of the circular loop
NUM_LOOP_POINTS = 500  # Number of points along the loop

# Calculation parameters
NK = 200  # Number of k-points for band structure and Berry phase calculations

# ============================================================================
# Create model with specified parameters
# ============================================================================
model = BerryBandModel(E_p=E_P, Delta=INITIAL_DELTA, t=T, 
                       delta=INITIAL_DELTA_SMALL, a=A, 
                       orbital_type=ORBITAL_TYPE)

# Calculate eigensystem
k_values, EiVa, EiVec = model.compute_eigensystem(NK)

# Calculate Berry phase for initial parameters
print("Calculating Berry phase for initial parameters...")
print(f"  Delta = {INITIAL_DELTA}, delta = {INITIAL_DELTA_SMALL}")
berry_phase_b1, berry_phase_b2 = model.calculate_berry_phase(EiVec, k_values)
print(f"  Berry phase Band 1: {berry_phase_b1:.6f} ")
print(f"  Berry phase Band 2: {berry_phase_b2:.6f} ")
print()

# Plot 1: Energy bands
print("Plotting energy bands...")
fig1, ax1 = plot_energy_bands(k_values, EiVa, title="Band structure")
plt.show()

# Plot 1b: h(k) ellipse in parameter space
print("Plotting h(k) ellipse...")
plot_h_ellipse(model.t, model.delta, model.Delta)

# Plot 1c: h(k) trajectory in 3D
print("Plotting h(k) trajectory in 3D...")
plot_h_ellipse_3d(model.t, model.delta, model.Delta)

# Create closed loop in parameter space
print("Creating closed loop...")
loop = create_closed_loop_circle(LOOP_CENTER, LOOP_RADIUS, NUM_LOOP_POINTS)

# Define key points to highlight (e.g., 4 evenly spaced points)
# These will be labeled A, B, C, D in all plots
NUM_KEY_POINTS = 4
key_point_indices = [i * (NUM_LOOP_POINTS // NUM_KEY_POINTS) for i in range(NUM_KEY_POINTS)]
# Ensure indices are within valid range
key_point_indices = [idx for idx in key_point_indices if idx < NUM_LOOP_POINTS]

# Plot 2: Closed loop with key points
print("Plotting closed loop with key points...")
fig2, ax2 = plot_closed_loop(loop, 
                             title=r"Closed Loop in $\Delta$-$\delta$ Space",
                             key_point_indices=key_point_indices)
plt.show()


# Calculate Berry phases along the loop
print("Calculating Berry phases along loop...")
results = calculate_berry_phases_for_parameters(model, loop, nk=NK)

# Plot 3: Berry phases with key points
print("Plotting Berry phases with key points...")
fig3, ax3 = plot_berry_phases(results, 
                              title="Berry Phase along Closed Loop",
                              key_point_indices=key_point_indices)
plt.show() 

# Plot 4: Evolution of h(k) trajectories in 3D with key points
print("Plotting evolution of h(k) trajectories in 3D with key points...")
# Plot every 10th point to avoid overcrowding (adjust step as needed)
fig4, ax4 = plot_h_trajectories_3d(model, loop, nk=200, step=10,
                                   key_point_indices=key_point_indices)
plt.show()

# Plot 5: h(k) trajectories in 3D for key points only (2x2 grid)
print("Plotting h(k) trajectories in 3D for key points...")
fig5, axes5 = plot_key_points_h_trajectories_3d(model, loop, key_point_indices, nk=200)
plt.show()

print("All plots completed!")

