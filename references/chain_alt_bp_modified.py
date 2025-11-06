"""
Reference script derived from Vanderbilt's PythTB examples:

- chain_alt.py    → builds the 1D chain with alternating hoppings / onsite terms
- chain_alt_bp.py → computes the Berry phase on a discretized 1D BZ
                    using the periodic-gauge factor e^{-i G r̂}

This version only:
  (i) wraps the original logic into functions,
  (ii) allows choosing the internal orbital positions explicitly,
  (iii) prints both the explicit Berry and PythTB's wf_array Berry
       to cross-check the implementation used in src/model.py.

This file is not imported by the main code.
Run it directly to reproduce Vanderbilt-style results.
"""

from pythtb import *
import matplotlib.pyplot as plt
import numpy as np
import math


def build_chain_model(t, del_t, Delta, orbital_positions):
    """
    Build a 1D SSH chain model with alternating site energies and hoppings.
    
    Parameters
    ----------
    t : float
        Average hopping strength.
    del_t : float
        Hopping alternation parameter.
    Delta : float
        Site energy alternation parameter.
        
    Returns
    -------
    tb_model
        PythTB model instance.
    """
    # 1D model with two orbitals per cell at positions [0.0, 0.5] (Onsite centered)
    lat = [[1.0]]
    orb = orbital_positions  # Each orbital is a separate list with its coordinate
    my_model = tb_model(1, 1, lat, orb)
    
    # Alternating site energies (average is zero)
    my_model.set_onsite([+Delta, -Delta])
    
    # Alternating hopping strengths
    my_model.set_hop(t + del_t, 0, 1, [0])   # hopping from orbital 0 to 1 within cell
    my_model.set_hop(t - del_t, 1, 0, [1])   # hopping from orbital 1 to 0 in next cell
    
    return my_model


def plot_chain_bands(model, nk=200, filename=None):
    """
    Plot the band structure of the chain model.
    
    Parameters
    ----------
    model : tb_model
        PythTB model instance.
    nk : int, optional
        Number of k-points along the path (default: 200).
    filename : str, optional
        Output filename for the plot. If None, plot is not saved.
        
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    # Construct the k-path
    (k_vec, k_dist, k_node) = model.k_path('fullc', nk)
    k_lab = (r'-$\pi$/2', r'0', r'$\pi$/2')
    
    # Solve for eigenvalues at each point on the path
    evals = model.solve_all(k_vec)
    
    # Set up the figure and specify details
    fig, ax = plt.subplots(figsize=(4., 3.))
    ax.set_xlim([0, k_node[-1]])
    ax.set_xticks(k_node)
    ax.set_xticklabels(k_lab)
    ax.axvline(x=k_node[1], linewidth=0.5, color='k')
    ax.set_xlabel("k")
    ax.set_ylabel("Band energy")
    
    # Plot first and second bands
    ax.plot(k_dist, evals[0], color='k')
    ax.plot(k_dist, evals[1], color='k')
    
    # Save figure if filename is provided
    fig.tight_layout()
    if filename is not None:
        fig.savefig(filename)
    
    return fig, ax


def berry_phase_explicit(model, nk=200, orbital_positions=None, band_index=0):
    """
    Compute Berry phase by explicit product of overlaps, following Vanderbilt (eq. 3.74).
    
    This function calculates the Berry phase by computing the product of overlaps
    between neighboring k-points along a closed path, and applying phase factors
    to close the periodic loop according to the orbital positions.
    
    Parameters
    ----------
    model : tb_model
        PythTB model instance.
    nk : int, optional
        Number of k-points along the closed path (default: 200).
    orbital_positions : array-like, optional
        Fractional positions of the two orbitals INSIDE THE CELL.
        Example:
            [0.0, 0.5]     -> site-centered (0, a/2)
            [-0.25, 0.25]  -> mid-bond centered (-a/4, +a/4)
        If None, uses the model's default orbital positions.
    band_index : int, optional
        Band index (0 or 1) for which to compute the Berry phase (default: 0).
        
    Returns
    -------
    float
        Berry phase in radians.
    """
    # Set up and solve the model on a discretized k mesh
    (k_vec, k_dist, k_node) = model.k_path('fullc', nk)
    (eval, evec) = model.solve_all(k_vec, eig_vectors=True)
    
    # Extract eigenvectors for the specified band
    evec_band = evec[band_index]  # shape: (nk, norb)
    
    # If orbital positions not specified, get them from the model
    if orbital_positions is None:
        orb_coords = model.get_orb()
        orbital_positions = [orb_coords[i][0] for i in range(len(orb_coords))]
    
    orbital_positions = np.array(orbital_positions)
    
    # Compute product of overlaps between neighboring k-points
    prod = 1. + 0.j
    for i in range(1, nk - 1):  # <u(k_0)|u(k_1)>...<u(k_{nk-2})|u(k_{nk-1})>
        prod *= np.vdot(evec_band[i - 1], evec_band[i])
    
    # Now compute the phase factors needed for last inner product
    # Apply periodic gauge: u(k+G) = exp(-i G·r) u(k)
    phase = np.exp((-2.j) * np.pi * orbital_positions)
    evec_last = phase * evec_band[0]  # evec[k_nk] constructed from evec[k_0]
    
    # Include the last overlap <u(k_{nk-1})|u(k_nk)>
    prod *= np.vdot(evec_band[-2], evec_last)
    
    # Berry phase is the negative argument of the product
    berry_phase = -np.angle(prod)
    
    return berry_phase


def berry_phase_pythtb(model, nk=200):
    """
    Compute Berry phase using PythTB's wf_array method.
    
    Parameters
    ----------
    model : tb_model
        PythTB model instance.
    nk : int, optional
        Number of k-points along the closed path (default: 200).
        
    Returns
    -------
    berry_phase_b1 : float
        Berry phase of band 0 in radians.
    berry_phase_b2 : float
        Berry phase of band 1 in radians.
    """
    # Create wf_array and solve on grid
    evec_array = wf_array(model, [nk])
    evec_array.solve_on_grid([0.])
    
    # Compute Berry phase for each band
    berry_phase_b1 = evec_array.berry_phase([0])  # Berry phase of bottom band
    berry_phase_b2 = evec_array.berry_phase([1])  # Berry phase of top band
    
    return berry_phase_b1, berry_phase_b2


if __name__ == "__main__":
    # Same parameters as original script
    Delta = 0
    t = -2.8
    del_t = 0
    
    # Build the model
    model_site = build_chain_model(t, del_t, Delta, orbital_positions=[[0.0], [0.5]])
    model_midbond = build_chain_model(t, del_t, Delta, orbital_positions=[[-0.25], [0.25]])
    model_site.display()
    model_midbond.display()
    
    print(f"number of orbitals: {model_site.get_num_orbitals()}")
    print(f"reduced coordinates of orbitals in format [orbital,coordinate.]: {model_site.get_orb()}")
    print(f"lattice vectors in format [vector,coordinate]: {model_site.get_lat()}")
    print(f"hamiltoniano k=0: {model_site._gen_ham(0.5)}")  

    print(f"number of orbitals: {model_midbond.get_num_orbitals()}")
    print(f"reduced coordinates of orbitals in format [orbital,coordinate.]: {model_midbond.get_orb()}")
    print(f"lattice vectors in format [vector,coordinate]: {model_midbond.get_lat()}")
    print(f"hamiltoniano k=0: {model_midbond._gen_ham(0.5)}")
    
    # Visualize model structure
    (fig, ax) = model_site.visualize(0)
    ax.set_title("Title goes here")
    fig.savefig("model_site.pdf")
    (fig, ax) = model_midbond.visualize(0)
    ax.set_title("Title goes here")
    fig.savefig("model_midbond.pdf")
    
    # Plot band structure
    plot_chain_bands(model_site, nk=200, filename="chain_alt_site.pdf")
    plot_chain_bands(model_midbond, nk=200, filename="chain_alt_midbond.pdf")
    
    # Explicit Berry phase calculation, two different gauges:
    # Band 0 with site-centered orbitals (0, a/2)
    bp_b1_site = berry_phase_explicit(model_site, nk=200, orbital_positions=[0.0, 0.5], band_index=0)
    # print(f"final product band 1: {np.exp(1j * bp_b1_site)}")
    print("Berry phase for band 1 (site-centered) is %7.3f" % bp_b1_site)
    
    # Band 1 with ste-centered orbitals (0, a/2)
    bp_b2_site = berry_phase_explicit(model_site, nk=200, orbital_positions=[0.0,0.5], band_index=1)
    # print(f"final product band 2: {np.exp(1j * bp_b2_site)}")
    print("Berry phase for band 2 (site-centered) is %7.3f" % bp_b2_site)
    
  # Band 0 with site-centered orbitals (0, a/2)
    bp_b1_midbond = berry_phase_explicit(model_midbond, nk=200, orbital_positions=[-0.25, 0.25], band_index=0)
    # print(f"final product band 1: {np.exp(1j * bp_b1_site)}")
    print("Berry phase for band 1 (midbond-centered) is %7.3f" % bp_b1_midbond)
    
    # Band 1 with ste-centered orbitals (0, a/2)
    bp_b2_midbond = berry_phase_explicit(model_midbond, nk=200, orbital_positions=[-0.25, 0.25], band_index=1)
    # print(f"final product band 2: {np.exp(1j * bp_b2_site)}")
    print("Berry phase for band 2 (midbond-centered) is %7.3f" % bp_b2_midbond)

    # PythTB wf_array method
    bp1_wf, bp2_wf = berry_phase_pythtb(model_site, nk=200)
    print("Berry phase of band 1 (wf_array) is %7.3f" % bp1_wf)
    print("Berry phase of band 2 (wf_array) is %7.3f" % bp2_wf)
