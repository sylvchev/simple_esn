# Simple_ESN

[![Coverage Status](https://coveralls.io/repos/sylvchev/simple_esn/badge.svg?branch=master&service=github)](https://coveralls.io/github/sylvchev/simple_esn?branch=master)
[![Travis CI](https://travis-ci.org/sylvchev/simple_esn.svg?branch=master)](https://travis-ci.org/sylvchev/simple_esn)
[![Code Climate](https://codeclimate.com/github/sylvchev/simple_esn/badges/gpa.svg)](https://codeclimate.com/github/sylvchev/simple_esn)

## Simple ESN

**Simple_esn** implement a Python class of simple Echo State Networks models
witin the Scikit-learn framework. It is intended to be a fast-and-easy
transformation of an input signal in a reservoir of neurons. The classification
or regression could be done with any scikit-learn classifier/regressor.

The `SimpleESN` object could be part of a `Pipeline` and its parameter space could
be explored with a `GridSearchCV` for example.

The code is inspired by the "minimalistic ESN example" proposed by Mantas
Lukoševičius. It is licenced under GPLv3.

## Useful links

-   Code from Mantas Lukoševičius: http://organic.elis.ugent.be/software/minimal
-   Code from Mantas Lukoševičius: http://minds.jacobs-university.de/mantas/code
-   More serious reservoir computing softwares: http://organic.elis.ugent.be/software
-   Scikit-learn, indeed: http://scikit-learn.org/

## Dependencies

The only dependencies are scikit-learn, numpy and scipy.

No installation is required.

## Example

Using the SimpleESN class is easy as:

```python
from simple_esn import SimpleESN
import numpy as np
n_samples, n_features = 10, 5
np.random.seed(0)
X = np.random.randn(n_samples, n_features)
esn = SimpleESN(n_readout = 2)
echoes = esn.fit_transform(X)
```

It could also be part of a Pipeline:

```python
from simple_esn import SimpleESN
# Pick your classifier
pipeline = Pipeline([('esn', SimpleESN(n_readout=1000)),
                     ('svr', svm.SVR())])
parameters = {
    'esn__weight_scaling': [0.5, 1.0],
    'svr__C': [1, 10]
}
grid_search = GridSearchCV(pipeline, parameters)
grid_search.fit(X_train, y_train)
```
