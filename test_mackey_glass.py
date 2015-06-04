"""This example file shows how to use the SimpleESN class, in the scikit-learn
fashion. It is inspired by the minimalistic ESN example of Mantas Lukoševičius
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

from simple_esn import SimpleESN
from sklearn.linear_model import Ridge
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from numpy import loadtxt, atleast_2d
import matplotlib.pyplot as plt
from pprint import pprint
from time import time
import numpy as np

if __name__ == '__main__':
    X = loadtxt('MackeyGlass_t17.txt')
    X = atleast_2d(X).T
    train_length = 2000
    test_length = 2000
    
    X_train = X[:train_length]
    y_train = X[1:train_length+1]
    X_test = X[train_length:train_length+test_length]
    y_test = X[train_length+1:train_length+test_length+1]

    # Simple training
    my_esn = SimpleESN(n_readout=1000, n_components=1000,
                       damping = 0.3, weight_scaling = 1.25)
    echo_train = my_esn.fit_transform(X_train)
    regr = Ridge(alpha = 0.01)
    regr.fit(echo_train, y_train)
    echo_test = my_esn.transform(X_test)
    y_true, y_pred = y_test, regr.predict(echo_test)
    err = mean_squared_error(y_true, y_pred)
    
    fp = plt.figure(figsize=(12, 4))
    trainplot = fp.add_subplot(1, 3, 1)
    trainplot.plot(X_train[100:600], 'b')
    trainplot.set_title('Some training signal')
    echoplot = fp.add_subplot(1, 3, 2)
    echoplot.plot(echo_train[100:600,:20])
    echoplot.set_title('Some reservoir activation')
    testplot =  fp.add_subplot(1, 3, 3)
    testplot.plot(X_test[-500:], 'b', label='test signal')
    testplot.plot(y_pred[-500:], 'g', label='prediction')
    testplot.set_title('Prediction (MSE %0.3f)' % err)
    testplot.legend(loc='lower right')
    plt.tight_layout(0.5)

    # Grid search
    pipeline = Pipeline([('esn', SimpleESN(n_readout=1000)),
                         ('ridge', Ridge(alpha = 0.01))])
    parameters = {
        'esn__n_readout': [1000],
        'esn__n_components': [1000],
        'esn__weight_scaling': [0.9, 1.25],
        'esn__damping': [0.3],
        'ridge__alpha': [0.01, 0.001]
    }
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, cv=3)
    print ("Starting grid search with parameters")
    pprint (parameters)
    t0 = time()
    grid_search.fit(X_train, y_train)
    print ("done in %0.3f s" % (time()-t0))
    
    print ("Best score on training is: %0.3f" % grid_search.best_score_)
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print ("\t%s: %r" % (param_name, best_parameters[param_name]))
    
    y_true, y_pred = y_test, grid_search.predict(X_test)
    err = mean_squared_error(y_true, y_pred)
    
    fg = plt.figure(figsize=(9, 4))
    echoplot = fg.add_subplot(1, 2, 1)
    echoplot.plot(echo_train[100:600,:20])
    echoplot.set_title('Some reservoir activation')
    testplot =  fg.add_subplot(1, 2, 2)
    testplot.plot(X_test[-500:], 'b', label='test signal')
    testplot.plot(y_pred[-500:], 'g', label='prediction')
    testplot.set_title('Prediction after GridSearch (MSE %0.3f)' % err)
    testplot.legend(loc='lower right')
    plt.tight_layout(0.5)
    plt.show()
    
    
