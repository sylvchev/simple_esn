import numpy as np
from numpy.testing import assert_array_equal
from nose.tools import assert_true
from simple_esn import SimpleESN

n_samples, n_features, n_readout = 10, 5, 2
rng_global = np.random.RandomState(0)
X = rng_global.randn(n_samples, n_features)

def test_SimpleESN_shape():
    esn =SimpleESN(n_readout = n_readout)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_readout)) 

def test_SimpleESN_discard():
    discard_steps = 3
    esn =SimpleESN(n_readout = n_readout, discard_steps = discard_steps)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples-discard_steps, n_readout)) 

def test_SimpleESN_incorrect_readout():
    n_readout, n_components = 10, 5
    esn =SimpleESN(n_readout = n_readout, n_components = n_components)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_components)) 

def test_SimpleESN_initialization():
    esn = SimpleESN(n_readout, n_components=100, damping=0.5,
                    weight_scaling=0.9, discard_steps=0, random_state=None)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_readout))

def test_SimpleESN_fit():
    esn = SimpleESN(n_readout = n_readout)
    esn = esn.fit(X)
    assert_true(esn.weights_ is not None)     
    assert_true(esn.input_weights_ is not None)     
    assert_true(esn.readout_idx_ is not None)
    
def test_SimpleESN_transform():
    esn =SimpleESN(n_readout = n_readout)
    echoes = esn.transform(X)
    assert_true(esn.weights_ is not None)     
    assert_true(esn.input_weights_ is not None)     
    assert_true(esn.readout_idx_ is not None)
    
    repeated_echoes = esn.transform(X)
    assert_array_equal (echoes, repeated_echoes)

    
