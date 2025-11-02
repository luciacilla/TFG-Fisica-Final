"""
SSH model with Berry phase calculation.

This module implements the Su-Schrieffer-Heeger (SSH) model and Berry phase
calculation using Vanderbilt's discrete formulation (2018).
"""

import numpy as np
import math
import matplotlib.pyplot as plt


class BerryBandModel:
    """
    SSH model for Berry phase calculation.
    
    This model implements a two-orbital per unit cell system with alternating
    hopping. The Berry phase is calculated using Vanderbilt's discrete formulation
    (Eq. 3.74), which requires applying phase factors depending on the internal
    positions of the orbitals.
    
    Parameters
    ----------
    E_p : float, optional
        Average site energy (default: 0).
    Delta : float, optional
        Site energy alternation (default: 0).
    t : float, optional
        Average hopping value (default: -2.8).
    delta : float, optional
        Hopping alternation (default: 0).
    a : float, optional
        Lattice constant (default: 1).
    orbital_type : str, optional
        Unit cell type:
        - 'midbond': orbitals at (-a/4, +a/4) → phases (-π/2, +π/2)
        - 'site': orbitals at (0, a/2) → phases (0, π)
        (default: 'midbond')
    """
    
    def __init__(self, E_p=0, Delta=0, t=-2.8, delta=0, a=1, orbital_type='midbond'):
        self.E_p = E_p
        self.Delta = Delta
        self.t = t
        self.delta = delta
        self.a = a
        self.orbital_type = orbital_type

    def update_parameters(self, Delta, delta):
        """
        Update Delta and delta parameters of the model.
        
        Parameters
        ----------
        Delta : float
            New site energy alternation.
        delta : float
            New hopping alternation.
        """
        self.Delta = Delta
        self.delta = delta

    def Hamiltonian(self, k):
        """
        Generate the Hamiltonian matrix for a given k point.
        
        The matrix has the form:
            [E_p + Delta,     h*(k)     ]
            [  h(k),      E_p - Delta   ]
        
        where h(k) = 2*t*cos(ka/2) + 2i*delta*sin(ka/2).
        
        Parameters
        ----------
        k : float
            Crystal momentum in reduced units.
            
        Returns
        -------
        numpy.ndarray
            2x2 Hamiltonian matrix at point k.
        """
        return np.array([
            [self.E_p + self.Delta, 2 * self.t * math.cos(k * self.a / 2) + 2j * self.delta * math.sin(k * self.a / 2)],
            [2 * self.t * math.cos(k * self.a / 2) - 2j * self.delta * math.sin(k * self.a / 2), self.E_p - self.Delta]
        ])

    def create_k_values(self, nk):
        """
        Generate an array of k values from -π to π.
        
        Parameters
        ----------
        nk : int
            Number of k points to generate.
            
        Returns
        -------
        numpy.ndarray
            Array of k values uniformly distributed between -π and π.
        """
        return np.linspace(-math.pi, math.pi, nk)

    def get_eigenvalues_and_vectors(self, k_values):
        """
        Calculate eigenvalues and eigenvectors for a set of k values.
        
        For each k, diagonalizes the Hamiltonian and sorts eigenvalues
        from lowest to highest. Eigenvectors are returned sorted accordingly.
        
        Parameters
        ----------
        k_values : array_like
            Array of k values for which to calculate the spectrum.
            
        Returns
        -------
        EiVa : list
            List of arrays, each containing sorted eigenvalues for one k value.
        EiVec : list
            List of lists. EiVec[i] contains eigenvectors (as arrays)
            of band i for all k values.
            Format: EiVec[band][k_index] = complex 2D vector
        """
        num_bands = 2
        EiVa = []
        EiVec = [[] for _ in range(num_bands)]

        for k in k_values:
            eigenvalues, eigenvectors = np.linalg.eigh(self.Hamiltonian(k))
            eigenvectors = eigenvectors.T  # Transpose: now eigenvectors[i] is the i-th eigenvector
            (eigenvalues_ord, eigenvectors_ord) = self._nicefy_eig(eigenvalues, eigenvectors)
            EiVa.append(eigenvalues_ord)
            for i in range(num_bands):
                EiVec[i].append(eigenvectors_ord[i, :])

        return EiVa, EiVec

    def get_coefficients(self, k_values):
        """
        Extract coefficients (eigenvectors) for each k value and band.
        
        Parameters
        ----------
        k_values : array_like
            Array of k values.
            
        Returns
        -------
        numpy.ndarray
            Array of shape (num_bands, num_k, 2) with coefficients
            of each band at each k point.
        """
        _, EiVec = self.get_eigenvalues_and_vectors(k_values)
        coefficients = np.array(EiVec)  # Convert list of lists to a NumPy array for easier handling
        return coefficients

    @staticmethod
    def _nicefy_eig(eval, eig=None):
        """
        Sort eigenvalues and eigenvectors, converting to real numbers.
        
        Eigenvalues are sorted from lowest to highest. Eigenvectors
        are reordered accordingly.
        
        Note: after np.linalg.eigh and transposition, eig has the
        shape (num_bands, dim), where eig[i, :] is the eigenvector
        corresponding to eigenvalue eval[i]. This function maintains
        that structure after reordering.
        
        Parameters
        ----------
        eval : array_like
            Eigenvalues (may be complex, but only real parts are taken).
        eig : array_like, optional
            Eigenvectors in format (num_bands, dim) where eig[i, :] is
            the eigenvector of eigenvalue eval[i].
            
        Returns
        -------
        eval : numpy.ndarray
            Sorted eigenvalues (real part).
        eig : numpy.ndarray, optional
            Reordered eigenvectors in the same format as input.
        """
        eval = np.array(eval.real, dtype=float)
        args = eval.argsort()
        eval = eval[args]
        if eig is not None:
            # eig[:, args] reorders columns (eigenvectors) according to args
            # Note: this maintains the structure where eig[i, :] is the i-th eigenvector
            eig = eig[:, args]
            return (eval, eig)
        return eval

    def calculate_berry_phase(self, EiVec, k_values):
        """
        Calculate Berry phase using Vanderbilt's discrete formulation (2018).
        
        This function implements equation 3.74 of Vanderbilt, which calculates
        the Berry phase as the argument of the product of overlaps between
        neighboring eigenvectors along a closed path in k-space.
        
        The discrete implementation requires applying phase factors when closing
        the periodic loop, depending on the internal positions of the orbitals
        in the unit cell. This reflects the periodic gauge choice where:
        
        u_{n,k+G} = exp(-i G·r) u_{n,k}
        
        where G is a reciprocal lattice vector and r are the orbital positions.
        
        The Berry phase value depends on the unit cell choice:
        
        - orbital_type='midbond': orbitals at (-a/4, +a/4)
          → applied phases: (-π/2, +π/2)
          → corresponds to a unit cell centered on the bond
        
        - orbital_type='site': orbitals at (0, a/2)
          → applied phases: (0, π)
          → corresponds to a unit cell with orbitals on sites
        
        Parameters
        ----------
        EiVec : list
            List of lists with eigenvectors. EiVec[band][k_index] contains
            the eigenvector of band 'band' at k point with index 'k_index'.
        k_values : array_like
            Array of k values (used only to determine path length,
            explicit values are not used).
            
        Returns
        -------
        berry_phase_b1 : float
            Berry phase of the lower band (in radians, between -π and π).
        berry_phase_b2 : float
            Berry phase of the upper band (in radians, between -π and π).
        """
        prod_b1, prod_b2 = 1 + 0.0j, 1 + 0.0j
        band_1, band_2 = EiVec[0], EiVec[1]
        
        # Product of overlaps between neighboring eigenvectors
        for i in range(len(k_values) - 1):
            prod_b1 *= np.vdot(band_1[i], band_1[i + 1])
            prod_b2 *= np.vdot(band_2[i], band_2[i + 1])
        
        # Select orbital positions according to unit cell type
        if self.orbital_type == 'midbond':
            # Orbitals at (-a/4, +a/4) → phases (-π/2, +π/2)
            orb = np.array([-math.pi/2, math.pi/2])
        elif self.orbital_type == 'site':
            # Orbitals at (0, a/2) → phases (0, π)
            orb = np.array([0, math.pi])
        else:
            raise ValueError("orbital_type must be either 'midbond' or 'site'")
            
        # Apply phase factors to close the periodic loop
        # These factors come from the periodic gauge condition:
        # u_{n,k+G} = exp(-i G·r) u_{n,k}
        phase = np.exp(-1j * orb)
        evec_last_b1 = phase * band_1[0]
        evec_last_b2 = phase * band_2[0]
        
        # Calculate final overlap with first point (after applying phases)
        final_prod_b1 = prod_b1 * np.vdot(band_1[-1], evec_last_b1)
        final_prod_b2 = prod_b2 * np.vdot(band_2[-1], evec_last_b2)
        
        # Berry phase is the negative argument of the total product
        berry_phase_b1 = -np.angle(final_prod_b1)
        berry_phase_b2 = -np.angle(final_prod_b2)
        
        return berry_phase_b1, berry_phase_b2

    def compute_eigensystem(self, nk):
        """
        Helper method that calculates the complete eigenvalue-eigenvector system.
        
        This method is a convenience to shorten scripts that need to
        calculate k_values, eigenvalues, and eigenvectors together.
        
        Parameters
        ----------
        nk : int
            Number of k points.
            
        Returns
        -------
        k_values : numpy.ndarray
            Array of k values.
        EiVa : list
            Eigenvalues for each k.
        EiVec : list
            Eigenvectors for each k and band.
        """
        k_values = self.create_k_values(nk)
        EiVa, EiVec = self.get_eigenvalues_and_vectors(k_values)
        return k_values, EiVa, EiVec

    def plot_h_ellipse(self, nk=400):
        """
        Visualize the ellipse of vector h(k) = (h_x(k), h_y(k)) in parameter space.
        
        For the SSH model, the Hamiltonian can be written as:
        H(k) = (E_p + Delta) σ_0 + h_x(k) σ_x + h_y(k) σ_y + Delta σ_z
        
        where:
        - h_x(k) = 2*t*cos(ka/2)
        - h_y(k) = -2*delta*sin(ka/2)
        
        This function parametrizes k and plots the trajectory (h_x, h_y) in the
        plane, which forms an ellipse when delta ≠ 0.
        
        Parameters
        ----------
        nk : int, optional
            Number of points to parametrize the ellipse (default: 400).
            
        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure with the plot.
        ax : matplotlib.axes.Axes
            Axes of the plot.
        """
        theta = np.linspace(-np.pi, np.pi, nk)
        k_values = theta  # k = theta in units where a=1
        h_x = 2 * self.t * np.cos(k_values * self.a / 2)
        h_y = -2 * self.delta * np.sin(k_values * self.a / 2)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(h_x, h_y, 'b-', linewidth=1.5)
        ax.set_xlabel(r'$h_x(k) = 2t\cos(ka/2)$')
        ax.set_ylabel(r'$h_y(k) = -2\delta\sin(ka/2)$')
        ax.set_title(r'Trajectory of $\mathbf{h}(k)$ in parameter space')
        ax.grid(True, alpha=0.3)
        ax.axhline(0, color='k', linewidth=0.5)
        ax.axvline(0, color='k', linewidth=0.5)
        ax.set_aspect('equal')
        plt.tight_layout()
        return fig, ax
