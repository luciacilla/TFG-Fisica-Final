"""
Plotting functions for SSH model results.

This module contains visualization functions for energy bands, parameter
space loops, and Berry phase results.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
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
    
    ax.set_xlabel('k', fontsize=12)
    ax.set_ylabel('E(k)', fontsize=12)
    ax.set_xticks([-math.pi, 0, math.pi], [r'$-\pi/a$', '0', r'$\pi/a$'])
    ax.tick_params(axis='both', which='major', labelsize=11)
    ax.legend(fontsize=11)
    if title is None:
        ax.set_title('Energy Bands as a function of k', fontsize=13)
    else:
        ax.set_title(title, fontsize=13)
    ax.grid(True)
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_closed_loop(loop_points, title=None, ax=None, save_path=None, key_point_indices=None):
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
    key_point_indices : list of int, optional
        List of indices in loop_points to highlight with labels A, B, C, D, etc.
        If None, no key points are highlighted.
        
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
    Delta_values = np.array(Delta_values)
    delta_values = np.array(delta_values)
    
    # Create indices for coloring
    num_points = len(loop_points)
    indices = np.arange(num_points)
    
    # Plot the loop line
    ax.plot(Delta_values, delta_values, '-', color='gray', linewidth=1.5, alpha=0.5)
    
    # Plot points with colors based on index
    scatter = ax.scatter(Delta_values, delta_values, c=indices, cmap='viridis', 
                        s=30, zorder=10, edgecolors='black', linewidths=0.3)
    
    # Highlight key points with labels A, B, C, D, etc. (discrete style)
    if key_point_indices is not None:
        labels = [chr(65 + i) for i in range(len(key_point_indices))]  # A, B, C, D, ...
        for label_idx, point_idx in enumerate(key_point_indices):
            if 0 <= point_idx < num_points:
                # Plot highlighted point with smaller, more discrete marker
                ax.scatter(Delta_values[point_idx], delta_values[point_idx], 
                          c='darkred', s=80, marker='s', zorder=15, 
                          edgecolors='black', linewidths=0.8, alpha=0.7)
                # Add label (more discrete)
                ax.annotate(labels[label_idx], 
                           (Delta_values[point_idx], delta_values[point_idx]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=11, fontweight='normal', color='darkred',
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                   edgecolor='darkred', alpha=0.6, linewidth=0.8),
                           zorder=20)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Point Index')
    cbar.ax.tick_params(labelsize=9)
    
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
    
    
    plt.tight_layout()
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig, ax


def plot_berry_phases(results, title=None, ax=None, save_path=None, key_point_indices=None):
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
    key_point_indices : list of int, optional
        List of indices in results to highlight with labels A, B, C, D, etc.
        If None, no key points are highlighted.
        
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
    indices = np.array(range(len(berry_phases_band1)))
    num_points = len(indices)
    
    # Plot points with colors based on index - different colormaps for each band
    # Band 1: viridis (green-blue tones)
    scatter1 = ax.scatter(indices, berry_phases_band1, c=indices, cmap='viridis', 
                         s=30, label='Band 1', edgecolors='black', linewidths=0.3, zorder=10)
    # Band 2: plasma (purple-pink tones) - similar to viridis but different hue
    scatter2 = ax.scatter(indices, berry_phases_band2, c=indices, cmap='plasma', 
                         s=30, label='Band 2', marker='s', edgecolors='black', 
                         linewidths=0.3, zorder=10)
    
    # Highlight key points with vertical lines and labels on x-axis
    if key_point_indices is not None:
        labels = [chr(65 + i) for i in range(len(key_point_indices))]  # A, B, C, D, ...
        for label_idx, point_idx in enumerate(key_point_indices):
            if 0 <= point_idx < num_points:
                # Draw thin vertical line at key point
                ax.axvline(x=point_idx, color='gray', linestyle='--', 
                          linewidth=0.8, alpha=0.6, zorder=5)
    
    # Add colorbar using viridis (same as loop plot)
    cbar = plt.colorbar(scatter1, ax=ax, label='Point Index')
    cbar.ax.tick_params(labelsize=9)
    
    # Set x-axis labels to show key point labels
    if key_point_indices is not None:
        # Get all x-tick positions
        ax.set_xlabel('Index of Point')
        # Set custom x-ticks at key point positions with labels
        ax.set_xticks(key_point_indices)
        ax.set_xticklabels(labels)
    else:
        ax.set_xlabel('Index of Point')
    
    ax.set_ylabel('Berry Phase (radians)')
    ax.set_ylim([-4, 4])  # Fixed y-axis range
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


def plot_h_ellipse_3d(t, delta, Delta=0, nk=400, show_axes=True, ax=None, title=None):
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
    ax : matplotlib.axes.Axes, optional
        Axes object to plot on. If None, creates a new figure.
    title : str, optional
        Title for the subplot. If None, uses default title.
    
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
    
    # Create 3D plot or use existing axes
    if ax is None:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure
    
    # Plot the trajectory
    ax.plot(hx, hy, hz, color='black', label=r'$\vec{h}(k)$', linewidth=1.5)
    
    if show_axes:
        # Draw coordinate axes at origin - long and smooth in light gray
        axis_length = axis_limit * 1.0  # Full length
        ax.plot([-axis_limit, axis_limit], [0, 0], [0, 0], 
                color='gray', linewidth=1.2, alpha=0.7, label=r'$h_x$ axis')
        ax.plot([0, 0], [-axis_limit, axis_limit], [0, 0], 
                color='gray', linewidth=1.2, alpha=0.7, label=r'$h_y$ axis')
        ax.plot([0, 0], [0, 0], [-axis_limit, axis_limit], 
                color='gray', linewidth=1.2, alpha=0.7, label=r'$h_z$ axis')
    
    # Set equal aspect ratio for all axes
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    
    # Set labels with larger font sizes
    ax.set_xlabel(r'$h_x$', fontsize=13)
    ax.set_ylabel(r'$h_y$', fontsize=13)
    ax.set_zlabel(r'$h_z$', fontsize=13)
    
    if title is None:
        ax.set_title(r'3D Trajectory of $\vec{h}(k)$', fontsize=14)
    else:
        ax.set_title(title, fontsize=12)
    
    # Improve tick labels
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=11)
    ax.tick_params(axis='z', labelsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add legend with larger font (only if ax is None, to avoid clutter in subplots)
    if ax is None:
        ax.legend(loc='upper right', fontsize=12)
    
    # Set background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return fig, ax


def plot_h_trajectories_3d(model, parameter_pairs, nk=200, step=1, title=None, key_point_indices=None):
    """
    Plot 3D trajectories of h(k) vector for multiple parameter pairs.
    
    This function shows how the h(k) trajectory evolves in 3D space as
    we move through different (Delta, delta) parameter pairs, typically
    along a closed loop.
    
    Parameters
    ----------
    model : BerryBandModel
        Instance of BerryBandModel (parameters will be updated for each pair).
    parameter_pairs : list of tuples
        List of (Delta, delta) pairs to plot trajectories for.
    nk : int, optional
        Number of k-points for each trajectory (default: 200).
    step : int, optional
        Plot every 'step' parameter pair (default: 1, plots all).
        Use step > 1 to reduce number of trajectories shown.
    title : str, optional
        Plot title. If None, uses default title.
    key_point_indices : list of int, optional
        List of indices in parameter_pairs to highlight with labels A, B, C, D, etc.
        The corresponding trajectories will be highlighted. If None, no key points are highlighted.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the 3D plot.
    ax : matplotlib.axes.Axes
        3D axes object.
    """
    # Create k values (same as plot_h_ellipse_3d)
    theta = np.linspace(-math.pi, math.pi, nk)
    
    # Create 3D plot
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get colormap for coloring trajectories based on parameter pair index
    num_pairs = len(parameter_pairs)
    colors = cm.viridis(np.linspace(0, 1, num_pairs))
    
    # Calculate max ranges for axis limits (same method as plot_h_ellipse_3d)
    hx_max_all = []
    hy_max_all = []
    hz_max_all = []
    
    for Delta, delta in parameter_pairs:
        model.update_parameters(Delta, delta)
        # Use same calculation as plot_h_ellipse_3d (t, delta, Delta are parameters)
        hx_temp = 2 * model.t * np.cos(theta)
        hy_temp = -2 * model.delta * np.sin(theta)
        hz_temp = np.full_like(theta, model.Delta)
        hx_max_all.append(np.max(np.abs(hx_temp)))
        hy_max_all.append(np.max(np.abs(hy_temp)))
        hz_max_all.append(np.max(np.abs(hz_temp)))
    
    max_range = max(max(hx_max_all), max(hy_max_all), max(hz_max_all))
    padding = max_range * 0.1 if max_range > 0 else 1.0
    axis_limit = max_range + padding
    
    # Plot trajectory for each parameter pair
    for idx, (Delta, delta) in enumerate(parameter_pairs[::step]):
        # Update model parameters
        model.update_parameters(Delta, delta)
        
        # Calculate h components (same equations as plot_h_ellipse_3d)
        hx = 2 * model.t * np.cos(theta)
        hy = -2 * model.delta * np.sin(theta)
        hz = np.full_like(theta, model.Delta)
        
        # Plot trajectory with color based on index
        original_idx = idx * step  # Original index in parameter_pairs
        color = colors[original_idx]
        ax.plot(hx, hy, hz, color=color, linewidth=1.0, alpha=0.6)
    
    # Set equal aspect ratio for all axes
    ax.set_xlim([-axis_limit, axis_limit])
    ax.set_ylim([-axis_limit, axis_limit])
    ax.set_zlim([-axis_limit, axis_limit])
    
    # Set labels
    ax.set_xlabel(r'$h_x$', fontsize=13)
    ax.set_ylabel(r'$h_y$', fontsize=13)
    ax.set_zlabel(r'$h_z$', fontsize=13)
    if title is None:
        num_shown = len(parameter_pairs[::step])
        ax.set_title(r'Evolution of $\vec{h}(k)$ trajectories' + f'\n({num_shown} parameter pairs)', fontsize=14)
    else:
        ax.set_title(title, fontsize=14)
    
    # Improve tick labels
    ax.tick_params(axis='x', labelsize=10)
    ax.tick_params(axis='y', labelsize=10)
    ax.tick_params(axis='z', labelsize=10)
    
    # Add grid
    ax.grid(True, alpha=0.25, linestyle='--')
    
    # Set background panes
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('gray')
    ax.yaxis.pane.set_edgecolor('gray')
    ax.zaxis.pane.set_edgecolor('gray')
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    
    # Draw axes in light gray
    ax.plot([-axis_limit, axis_limit], [0, 0], [0, 0], 
            color='gray', linewidth=1.2, alpha=0.7)
    ax.plot([0, 0], [-axis_limit, axis_limit], [0, 0], 
            color='gray', linewidth=1.2, alpha=0.7)
    ax.plot([0, 0], [0, 0], [-axis_limit, axis_limit], 
            color='gray', linewidth=1.2, alpha=0.7)
    
    # Add colorbar for parameter pair index
    sm = plt.cm.ScalarMappable(cmap=cm.viridis, 
                               norm=plt.Normalize(vmin=0, vmax=num_pairs-1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label='Parameter Pair Index', pad=0.1)
    cbar.ax.tick_params(labelsize=9)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax


def plot_key_points_h_trajectories_3d(model, parameter_pairs, key_point_indices, nk=200, 
                                      show_axes=True, title=None, save_path=None):
    """
    Plot 3D trajectories of h(k) for key points in a 2x2 subplot grid.
    
    Uses plot_h_ellipse_3d for each key point to calculate and display
    the trajectories in a 2x2 grid layout.
    
    Parameters
    ----------
    model : BerryBandModel
        Instance of BerryBandModel (t parameter will be used).
    parameter_pairs : list of tuples
        List of (Delta, delta) pairs representing the loop.
    key_point_indices : list of int
        List of indices in parameter_pairs to plot. Should contain 4 indices
        for a 2x2 grid. If more than 4, only the first 4 will be plotted.
    nk : int, optional
        Number of k-points for each trajectory (default: 200).
    show_axes : bool, optional
        Whether to draw the coordinate axes in each subplot (default: True).
    title : str, optional
        Overall figure title. If None, uses default title.
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
        
    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object containing the 2x2 grid of 3D plots.
    axes : numpy.ndarray
        Array of 3D axes objects (2x2).
    """
    # Limit to 4 points for 2x2 grid
    num_plots = min(4, len(key_point_indices))
    key_indices = key_point_indices[:num_plots]
    labels = [chr(65 + i) for i in range(num_plots)]  # A, B, C, D
    
    # Create figure with 2x2 subplots
    fig = plt.figure(figsize=(16, 16))
    axes = []
    
    # Calculate global axis limits for consistency across all subplots
    theta = np.linspace(-math.pi, math.pi, nk)
    hx_max_all = []
    hy_max_all = []
    hz_max_all = []
    
    for idx in key_indices:
        if 0 <= idx < len(parameter_pairs):
            Delta, delta = parameter_pairs[idx]
            # Use same calculation as plot_h_ellipse_3d
            hx_temp = 2 * model.t * np.cos(theta)
            hy_temp = -2 * delta * np.sin(theta)
            hz_temp = np.full_like(theta, Delta)
            hx_max_all.append(np.max(np.abs(hx_temp)))
            hy_max_all.append(np.max(np.abs(hy_temp)))
            hz_max_all.append(np.max(np.abs(hz_temp)))
    
    max_range = max(max(hx_max_all), max(hy_max_all), max(hz_max_all))
    padding = max_range * 0.1 if max_range > 0 else 1.0
    axis_limit = max_range + padding
    
    # Create subplots and use plot_h_ellipse_3d for each
    for i, (label, idx) in enumerate(zip(labels, key_indices)):
        if 0 <= idx < len(parameter_pairs):
            Delta, delta = parameter_pairs[idx]
            
            # Create 3D subplot
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            axes.append(ax)
            
            # Use plot_h_ellipse_3d with the subplot axis
            subplot_title = f'Point {label}: Δ={Delta:.3f}, δ={delta:.3f}'
            _, _ = plot_h_ellipse_3d(model.t, delta, Delta, nk=nk, 
                                     show_axes=show_axes, ax=ax, title=subplot_title)
            
            # Override axis limits for consistency across all subplots
            ax.set_xlim([-axis_limit, axis_limit])
            ax.set_ylim([-axis_limit, axis_limit])
            ax.set_zlim([-axis_limit, axis_limit])
    
    # Set overall figure title
    if title is None:
        fig.suptitle(r'3D Trajectories of $\vec{h}(k)$ for Key Points', fontsize=14, y=0.98)
    else:
        fig.suptitle(title, fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    if save_path is not None:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    return fig, np.array(axes).reshape(2, 2)

