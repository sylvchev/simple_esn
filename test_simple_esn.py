import numpy as np

from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_true
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import ignore_warnings

from simple_esn import SimpleESN

n_samples, n_features = 10, 5
rng_global = np.random.RandomState(0)
X = rng_global.randn(n_samples, n_features)

def test_SimpleESN_shape():
    n_readout = 2
    esn =SimpleESN(n_readout = n_readout)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_readout)) 

def test_SimpleESN_discard():
    n_readout, discard_steps = 2, 3
    esn =SimpleESN(n_readout = n_readout, discard_steps = discard_steps)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples-discard_steps, n_readout)) 

def test_SimpleESN_incorrect_readout():
    n_readout, n_components = 10, 5
    esn =SimpleESN(n_readout = n_readout, n_components = n_components)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_components)) 

def test_SimpleESN_initialization():
    n_readout = 2
    esn = SimpleESN(n_readout, n_components=100, damping=0.5,
                    weight_scaling=0.9, discard_steps=0, random_state=None)
    echoes = esn.fit_transform(X)
    assert_true(echoes.shape == (n_samples, n_readout)) 

def test_SimpleESN_fit_transform():
    # verifier que X et echoes sont differents
    pass

def test_SimpleESN_transform():
    # verifier que transform marche meme s'il n'y pas eu de fit avant
    
    # verifier que deux transformations sur un X produisent les memes echos
    pass

    
