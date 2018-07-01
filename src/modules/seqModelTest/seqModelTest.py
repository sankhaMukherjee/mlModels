from logs import logDecorator as lD 
from lib.sequentialModel import sequentialModel as sM

import json
import numpy      as np
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.modules.seqModelTest.seqModelTest'


@lD.log(logBase + '.testModel')
def testModel(logger):
    '''print a line
    
    This function simply prints a single line
    
    Parameters
    ----------
    logger : {[type]}
        [description]
    '''

    # Generate some data ...
    N = 1000
    print('Generating some data')
    X = np.random.rand(N, 2)
    y = 2*X[:, 0] + 3*X[:, 1] + 3 + 0.1*np.random.normal(0, 1, (N,))
    y = y.reshape(-1, 1)

    # Generate the regression model ...
    print('Generating an instance of the model')
    nInp        = 2
    layers      = [5, 6, 1]
    activations = [tf.tanh, tf.tanh, None]
    dropout     = [0.1, 0.1, 0.1]

    sReg = sM.SequentialRegress(nInp, layers, activations, dropout)

    # Lets fit the model
    print('Fitting the model')
    sReg.fit(X, y, N = 5000, show=100)

    # Predict some new data
    print('predicting data')
    yHat = sReg.predict(X)
    print(yHat.flatten()[:10])

    # Predict some new data
    print('predicting data')
    errVal = sReg.calcErr(X, y)
    print(errVal)

    return

@lD.log(logBase + '.main')
def main(logger):
    '''main function for module1
    
    This function finishes all the tasks for the
    main function. This is a way in which a 
    particular module is going to be executed. 
    
    Parameters
    ----------
    logger : {logging.Logger}
        The logger function
    '''

    testModel()

    return

