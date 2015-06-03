from simple_esn import SimpleESN
from sklearn import linear_model, svm
import numpy as np
import matplotlib.pyplot as plt

from scipy.linalg import inv
from numpy import eye, dot

if __name__ == '__main__':
    X = np.loadtxt('MackeyGlass_t17.txt')
    X = np.atleast_2d(X).T
    train_length = 2000
    test_length = 2000
    discard_steps = 100
    X_train = X[:train_length]
    y_train = X[discard_steps+1:train_length+1]
    X_test = X[train_length:train_length+test_length]

    my_esn = SimpleESN(n_readout=1000, n_components=1000, damping = 0.3, weight_scaling = 1.25, discard_steps=discard_steps)
    echo_train = my_esn.fit_transform(X_train)
    regr = linear_model.Ridge(alpha = 0.01) # regr = linear_model.LinearRegression(normalize=True) ou regr = linear_model.SGDRegressor(alpha=0.000001, n_iter=20, loss='epsilon_insensitive', penalty='l2')
    
    regr.fit(echo_train, y_train)
    reg = 1e-8  # regularization coefficient
    coef = dot(dot(y_train.T, echo_train), inv(dot(echo_train.T, echo_train)
                                               + reg*eye(1000)))

    echo_test = my_esn.transform(X_test)
    y_pred = regr.predict(echo_test)
    y_lsq = echo_test.dot(coef.T)

    fig = plt.figure(figsize=(15, 8))
    trainplot = fig.add_subplot(2, 3, 1)
    trainplot.plot(X_train[discard_steps:], 'b')
    trainplot.plot(y_train, 'g')
    trainplot.set_title('Training signal')
    tpaxis = trainplot.axis()

    traindiff = fig.add_subplot(2, 3, 4)
    traindiff.plot(X_train[discard_steps:]-y_train, 'g')
    traindiff.axis(tpaxis)
    traindiff.set_title('Baseline error')
    
    echoplot = fig.add_subplot(2, 3, (2, 5))
    echoplot.plot(echo_train[:,:100])
    echoplot.set_title('Some reservoir activation')

    testplot =  fig.add_subplot(2, 3, 3)
    testplot.plot(X_test[discard_steps:], 'b')
    testplot.plot(y_pred, 'g')
    testplot.plot(y_lsq, 'r')
    testplot.set_title('Test signal')
    testplot.axis(tpaxis)

    testdiff = fig.add_subplot(2, 3, 6)
    testdiff.plot(X_test[discard_steps:]-X[train_length+discard_steps+1:train_length+test_length+1], 'b')
    testdiff.plot(X[train_length+discard_steps+1:train_length+test_length+1]-y_pred, 'g')
    testdiff.plot(X[train_length+discard_steps+1:train_length+test_length+1]-y_lsq, 'r')
    testdiff.axis(tpaxis)
    testdiff.set_title('Prediction error')
    plt.show()
    
