"""Simple Echo State Network
"""

# Copyright (C) 2015 Sylvain Chevallier <sylvain.chevallier@uvsq.fr>

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# TODO: add n_readout = -1 for n_readout = n_components

from __future__ import print_function
import sys
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils import check_random_state, check_array
from numpy import zeros, ones, concatenate, array, tanh, vstack, arange
import numpy as np
import scipy.linalg as la

class SimpleESN(BaseEstimator, TransformerMixin):
    """Simple Echo State Network (ESN)

    Neuron reservoir of sigmoid units, with recurrent connection and random
    weights. Forget factor (or damping) ensure echoes in the network. No
    learning takes place in the reservoir, readout is left at the user's
    convience. The input processed by these ESN should be normalized in [-1, 1]

    Parameters
    ----------
    n_readout : int
        Number of readout neurons, chosen randomly in the reservoir. Determines
        the dimension of the ESN output.
    
    n_components : int, optional
        Number of neurons in the reservoir, 100 by default.

    damping : float, optional
        Damping (forget) factor for echoes, strong impact on the dynamic of the
        reservoir. Possible values between 0 and 1, default is 0.5

    weight_scaling : float, optional
        Spectral radius of the reservoir, i.e. maximum eigenvalue of the weight
        matrix, also strong impact on the dynamical properties of the reservoir.
        Classical regimes involve values around 1, default is 0.9

    discard_steps : int, optional
        Discard first steps of the timeserie, to allow initialisation of the
        network dynamics.

    random_state : integer or numpy.RandomState, optional
        Random number generator instance. If integer, fixes the seed.
        
    Attributes
    ----------
    input_weights_ : array_like, shape (n_features,)
        Weight of the input units

    weights_ : array_Like, shape (n_components, n_components)
        Weight matrix for the reservoir

    components_ : array_like, shape (n_samples, 1+n_features+n_components)
        Activation of the n_components reservoir neurons, including the
        n_features input neurons and the bias neuron, which has a constant
        activation.

    readout_idx_ : array_like, shape (n_readout,)
        Index of the randomly selected readout neurons

    Example
    -------

    >>> from simple_esn import SimpleESN
    >>> import numpy as np
    >>> n_samples, n_features = 10, 5
    >>> np.random.seed(0)
    >>> X = np.random.randn(n_samples, n_features)
    >>> esn =SimpleESN(n_readout = 2)
    >>> echoes = esn.fit_transform(X)
    """
    def __init__(self, n_readout, n_components=100, damping=0.5,
                 weight_scaling=0.9, discard_steps=0, random_state=None):
        self.n_readout = n_readout
        self.n_components = n_components
        self.damping = damping
        self.weight_scaling = weight_scaling
        self.discard_steps = discard_steps
        self.random_state = check_random_state(random_state)
        self.input_weights_ = None
        self.readout_idx_ = None
        self.weights_ = None

    def _fit_transform(self, X):
        n_samples, n_features = X.shape
        X = check_array(X, ensure_2d=True)
        self.weights_ = self.random_state.rand(self.n_components, self.n_components)-0.5
        spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
        self.weights_ *=  self.weight_scaling / spectral_radius
        self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        self.readout_idx_ = self.random_state.permutation(arange(1+n_features,
                                    1+n_features+self.n_components))[:self.n_readout]
        self.components_ = zeros(shape=(1+n_features+self.n_components,
                                        n_samples))

        curr_ = zeros(shape=(self.n_components, 1))
        U = concatenate((ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = array(U[t,:], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*tanh(
                self.input_weights_.dot(u) + self.weights_.dot(curr_))
            self.components_[:,t] = vstack((u, curr_))[:,0]
        return self

    def fit(self, X, y=None):
        """Initialize the network

        This is more compatibility step, as no learning takes place in the
        reservoir.
        
        Parameters
        ----------
        X : array-like shape, (n_samples, n_features)
            The data to be transformed.
            
        Returns
        -------
        self : returns an instance of self.
        """
        self = self._fit_transform(X)
        return self
    
    def fit_transform(self, X, y=None):
        """Generate echoes from the reservoir.

        Parameters
        ----------
        X : array-like of shape [n_samples, n_features]
            training set.

        Returns
        -------
        readout : array, shape (n_samples, n_readout)
            Reservoir activation generated by the readout neurons
        """
        self = self._fit_transform(X)
        return self.components_[self.readout_idx_, self.discard_steps:].T

    def transform(self, X):
        """Generate echoes from the reservoir

        Parameters
        ----------
        X : array-like shape, (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        readout : array, shape (n_samples, n_readout)
            Reservoir activation generated by the readout neurons
        """
        X = check_array(X, ensure_2d=True)
        n_samples, n_features = X.shape
        
        if self.weights_ is None:
            self.weights_ = self.random_state.rand(self.n_components,
                                                   self.n_components)-0.5
            spectral_radius = np.max(np.abs(la.eig(self.weights_)[0]))
            self.weights_ *=  self.weight_scaling / spectral_radius
        if self.input_weights_ is None:
            self.input_weights_ = self.random_state.rand(self.n_components,
                                                         1+n_features)-0.5
        if self.readout_idx_ is None:
            self.readout_idx_ = self.random_state.permutation(arange(1+n_features,
                                    1+n_features+self.n_components))[:self.n_readout]
        self.components_ = zeros(shape=(1+n_features+self.n_components,
                                        n_samples))

        curr_ = zeros(shape=(self.n_components, 1))
        U = concatenate((ones(shape=(n_samples, 1)), X), axis=1)
        for t in range(n_samples):
            u = array(U[t,:], ndmin=2).T
            curr_ = (1-self.damping)*curr_ + self.damping*tanh(
                self.input_weights_.dot(u) + self.weights_.dot(curr_))
            self.components_[:,t] = vstack((u, curr_))[:,0]
                
        return self.components_[self.readout_idx_, self.discard_steps:].T
