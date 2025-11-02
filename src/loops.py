"""
Utility functions for creating parameter loops and calculating Berry phases.

This module provides functions to generate closed loops in parameter space
and compute Berry phases along those loops.
"""

import numpy as np


def create_closed_loop_circle(center, radius, num_points):
    """
    Create a closed loop on a circle in the (Delta, delta) plane.
    
    Parameters
    ----------
    center : tuple
        The (x, y) coordinates of the circle's center, i.e., (Delta_0, delta_0).
    radius : float
        The radius of the circle.
    num_points : int
        Number of points to sample along the loop.
    
    Returns
    -------
    list
        List of (Delta, delta) tuples representing the loop.
        The loop is closed, with the last point coinciding with the first.
    """
    cx, cy = center
    points = []
    # Generate points from 0 to just under 2*pi
    # The last point is added manually to exactly complete the loop at 2*pi
    for i in range(num_points - 1):
        angle = 2 * np.pi * i / (num_points - 1)
        x = cx + radius * np.cos(angle)
        y = cy + radius * np.sin(angle)
        points.append((x, y))
    
    # Add the last point at exactly 2*pi to close the loop
    # This point technically overlaps with the first point at angle 0
    last_angle = 0
    x_last = cx + radius * np.cos(last_angle)
    y_last = cy + radius * np.sin(last_angle)
    points.append((x_last, y_last))

    return points


def calculate_berry_phases_for_parameters(model, pairs, nk):
    """
    Calculate Berry phases for a list of (Delta, delta) parameter pairs.
    
    IMPORTANT: This function uses the original calculate_berry_phase method
    from the model class. It does NOT reimplement the Berry phase calculation.
    
    Parameters
    ----------
    model : BerryBandModel
        An instance of BerryBandModel.
    pairs : list of tuples
        List of (Delta, delta) pairs to evaluate.
    nk : int
        Number of k-points for the Berry phase calculation.
        
    Returns
    -------
    list
        A list of tuples with (Delta, delta, berry_phase_b1, berry_phase_b2).
    """
    results = []
    k_values = model.create_k_values(nk)
    
    for Delta, delta in pairs:
        # Update model parameters
        model.update_parameters(Delta, delta)
        
        # Calculate eigenvectors
        EiVa, EiVec = model.get_eigenvalues_and_vectors(k_values)
        
        # Calculate Berry phases using the model's method
        # This is the ORIGINAL implementation, not a reimplementation
        berry_phase_b1, berry_phase_b2 = model.calculate_berry_phase(EiVec, k_values)
        
        results.append((Delta, delta, berry_phase_b1, berry_phase_b2))
    
    return results

