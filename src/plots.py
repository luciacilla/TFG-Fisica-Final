"""
Plotting functions for SSH model results.

This module contains visualization functions for energy bands, parameter
space loops, and Berry phase results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


def plot_energy_bands(k_values, EiVa, title=None, ax=None, save_path=None):
    """
    Plot energy bands as a function of k.
    
    Parameters
    ----------
    k_values : array_like
        Array of k values.
    EiVa : list
        List of eigenvalue arrays for each k point.
    title : str, optional
        Plot title. If None, uses default title.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    num_bands = len(EiVa[0]) if len(EiVa) > 0 else 2
    
    for i in range(num_bands):
        energies = [eigenvalue[i] for eigenvalue in EiVa]
        ax.plot(k_values, energies, label=f'Band {i + 1}')
    
    ax.set_xlabel('k')
    ax.set_ylabel('E(k)')
    ax.set_xticks([-math.pi, 0, math.pi], [r'$-\pi/a$', '0', r'$\pi/a$'])
    ax.legend()
    if title is None:
        ax.set_title('Energy Bands as a function of k')
    else:
        ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_closed_loop(loop_points, title=None, ax=None, save_path=None):
    """
    Plot a closed loop in the (Delta, delta) parameter space.
    
    Parameters
    ----------
    loop_points : list of tuples
        List of (Delta, delta) points representing the loop.
    title : str, optional
        Plot title. If None, uses default title.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    else:
        fig = ax.figure
    
    # Unpack the list of tuples into separate lists
    Delta_values, delta_values = zip(*loop_points)
    
    # Plot the loop with points
    ax.plot(Delta_values, delta_values, '-o', markersize=3, linewidth=1.5)
    
    # Set axis labels
    ax.set_xlabel(r'$\Delta$')
    ax.set_ylabel(r'$\delta$')
    
    # Set title
    if title is None:
        ax.set_title(r'Closed Loop in $\Delta$-${\delta}$ Space')
    else:
        ax.set_title(title)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set equal scaling
    ax.set_aspect('equal')
    
    # Ensure the axes cross at (0,0)
    ax.axhline(0, color='black', linewidth=0.8)  # Horizontal line at y=0
    ax.axvline(0, color='black', linewidth=0.8)  # Vertical line at x=0
    
    # Mark the origin with a black point
    ax.scatter(0, 0, color='black', s=50, zorder=10)  # s is the size of the point
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_berry_phases(results, title=None, ax=None, save_path=None):
    """
    Plot Berry phases for each (Delta, delta) pair vs. point index.
    
    Parameters
    ----------
    results : list of tuples
        Output from calculate_berry_phases_for_parameters,
        each tuple contains (Delta, delta, berry_phase_b1, berry_phase_b2).
    title : str, optional
        Plot title. If None, uses default title.
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object.
    ax : matplotlib.axes.Axes
        Axes object.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = ax.figure
    
    # Unpack results
    _, _, berry_phases_band1, berry_phases_band2 = zip(*results)
    
    # Prepare index for each point
    indices = range(len(berry_phases_band1))
    
    # Plot for band 1
    ax.plot(indices, berry_phases_band1, '-o', label='Band 1', color='blue', markersize=4)
    
    # Plot for band 2
    ax.plot(indices, berry_phases_band2, '-o', label='Band 2', color='green', markersize=4)
    
    ax.set_xlabel('Index of Point')
    ax.set_ylabel('Berry Phase (radians)')
    if title is None:
        ax.set_title('Berry Phase for each Band over Points')
    else:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax

def plot_h_ellipse(t, delta, Delta=0, nk=400, show_axes=True):
    """
    Plot the full geometric trajectory of the SSH model vector h(θ)
    in the (h_x, h_y) plane, showing the complete closed ellipse.

    Parameters
    ----------
    t : float
        Average hopping amplitude.
    delta : float
        Dimerization parameter.
    Delta : float, optional
        On-site potential difference (plotted as a color height if ≠ 0).
    nk : int, optional
        Number of sampling points for θ. Default is 400.
    show_axes : bool, optional
        Whether to draw the coordinate axes. Default True.

    Notes
    -----
    In this model, the Hamiltonian components are:
        h_x = 2 t cos(θ)
        h_y = -2 δ sin(θ)
        h_z = Δ  (constant)
    The ellipse is fully traced when θ ∈ [−π, π].
    """
    theta = np.linspace(-math.pi, math.pi, nk)

    hx = 2 * t * np.cos(theta)
    hy = -2 * delta * np.sin(theta)

    # Calculate axis limits with consistent padding
    hx_max = np.max(np.abs(hx))
    hy_max = np.max(np.abs(hy))
    max_range = max(hx_max, hy_max)
    padding = max_range * 0.1  # 10% padding
    
    # Set symmetric limits
    xlim = [-max_range - padding, max_range + padding]
    ylim = [-max_range - padding, max_range + padding]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(hx, hy, label=r'$\vec{h}(\theta)$', linewidth=1.5)

    if show_axes:
        ax.axhline(0, color='black', lw=0.8, zorder=0)
        ax.axvline(0, color='black', lw=0.8, zorder=0)

    # Set equal aspect ratio and consistent axis limits
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    # Ensure axes cross at origin
    ax.set_xlabel(r'$h_x = 2 t \cos\theta$')
    ax.set_ylabel(r'$h_y = -2 \delta \sin\theta$')
    ax.set_title(r'Trajectory of $\vec{h}(\theta)$ (full ellipse)')
    ax.grid(True, ls=':', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_h_ellipse_3d(t, delta, Delta=0, nk=400, show_axes=True):
    """
    Plot the full 3D trajectory of the SSH model vector h(θ) in the
    (h_x, h_y, h_z) space, showing the complete closed path with constant z component.
    
    This function visualizes how the Hamiltonian vector h(θ) traces out a curve
    in three-dimensional parameter space. The trajectory forms an ellipse in the
    (h_x, h_y) plane (when viewed from above), while maintaining a constant
    component h_z = Δ along the z-axis.
    
    Parameters
    ----------
    t : float
        Average hopping amplitude.
    delta : float
        Dimerization parameter.
    Delta : float, optional
        On-site potential difference, which determines the constant h_z component
        (default: 0).
    nk : int, optional
        Number of sampling points for θ. Default is 400.
    show_axes : bool, optional
        Whether to draw the coordinate axes. Default True.
    
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the 3D plot.
    ax : matplotlib.axes.Axes
        3D axes object.
    
    Notes
    -----
    In the SSH model, the Hamiltonian components are:
        h_x(θ) = 2 t cos(θ)
        h_y(θ) = -2 δ sin(θ)
        h_z = Δ  (constant)
    
    The parameter θ ∈ [−π, π] traces a complete cycle, forming an ellipse
    in the (h_x, h_y) plane while maintaining constant h_z = Δ. When Δ = 0,
    the trajectory lies entirely in the (h_x, h_y) plane. When Δ ≠ 0, the
    trajectory is offset along the z-axis.
    
    The plot maintains equal aspect ratios on all axes to properly visualize
    the geometric structure of the trajectory.
    """
    theta = np.linspace(-math.pi, math.pi, nk)
    
    # Calculate h components
    hx = 2 * t * np.cos(theta)
    hy = -2 * delta * np.sin(theta)
    hz = np.full_like(theta, Delta)  # Constant z component
    
    # Calculate axis limits with consistent padding for all three axes
    hx_max = np.max(np.abs(hx))
    hy_max = np.max(np.abs(hy))
    hz_max = np.abs(Delta)
    
    # Use the maximum range to set consistent limits for all axes
    max_range = max(hx_max, hy_max, hz_max)
    padding = max_range * 0.1 if max_range > 0 else 1.0  # 10% padding, minimum 1.0
    
    # Set symmetric limits for all axes
    axis_limit = max_range + padding
    
    # Create 3D plot
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the trajectory
    ax.plot(hx, hy, hz, label=r'$\vec{h}(\theta)$', linewidth=1.5)
    
    if show_axes:
        # Draw coordinate axes at origin
        axis_length = axis_limit * 0.8
        ax.plot([0, axis_length], [0, 0], [0, 0], 'r-', linewidth=1.5, alpha=0.6, label=r'$h_x$ axis')
        ax.plot([0, 0], [0, axis_length], [0, 0], 'g-', linewidth=1.5, alpha=0.6, label=r'$h_y$ axis')
        ax.plot([0, 0], [0, 0], [0, axis_length], 'b-', linewidth=1.5, alpha=0.6, label=r'$h_z$ axis')
    
    # Set equal aspect ratio for all axes
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    
    # Set labels
    ax.set_xlabel(r'$h_x = 2t\cos\theta$', fontsize=11)
    ax.set_ylabel(r'$h_y = -2\delta\sin\theta$', fontsize=11)
    ax.set_zlabel(r'$h_z = \Delta$', fontsize=11)
    ax.set_title(r'3D Trajectory of $\vec{h}(\theta)$', fontsize=12)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

