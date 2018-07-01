
from tqdm     import tqdm
from logs     import logDecorator as lD
from datetime import datetime     as dt

import json, os
import numpy      as np
import tensorflow as tf

config = json.load(open('../config/config.json'))
logBase = config['logging']['logBase'] + '.lib.sequentialModel.sequentialModel'

class SequentialRegress():
    '''A simple sequential model for regression
    
    This is a simple sequential model. This will be used for
    training a simple deep learner.
    '''

    @lD.log(logBase + '.SequentialRegress.saveModel')
    def __init__(logger, self, nInp, layers, activations, dropout=None, name=''):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        nInp : {[type]}
            [description]
        layers : {[type]}
            [description]
        activations : {[type]}
            [description]
        dropout : {[type]}, optional
            [description] (the default is None, which [default_description])
        name : {str}, optional
            An optional name given to the model. This will be 
            augmented with the generated name, and can be used
            later for grouping different models.
        '''

        self.nInp          = nInp
        self.layers        = layers
        self.activations   = activations
        if dropout is None:
            self.dropout   = [0]*len(layers)
        else:
            self.dropout   = dropout
        self.name          = name
        self.restorePoints = []

        Nlay  = len(layers)
        Nact  = len(activations)
        Ndrop = -1 if dropout is not None else len(dropout)

        self.feed_dict = {
            'Misc/dropout:0'       : self.dropout,
            'Misc/isTraining:0'    : False,
            'Misc/learning_rate:0' : 0.01,
        }

        # Make sure that the lengths are correct
        assert ( Nlay == Nact, 
                    'activations (N={}) and layers(N={}) have different lengths'.format(
                        Nlay, Nact))
        if dropout is not None:
            assert ( Nlay == Nact, 
                    'dropout and layers have different lengths'.format(
                        Ndrop, Nact))

        # placeholders used for different tasks
        with tf.variable_scope('Misc'):
            self.dropoutNN  = tf.placeholder(dtype=tf.float32, shape=len(layers), name='dropout')
            self.isTraining = tf.placeholder(dtype=tf.bool, shape=(), name='isTraining')
            self.lr         = tf.placeholder(dtype=tf.float32, shape=(), name='learning_rate')

        with tf.variable_scope('IO'):
            self.inp = tf.placeholder(dtype=tf.float32, shape=(None, nInp ), name='inp')
            self.out = tf.placeholder(dtype=tf.float32, shape=(None, layers[-1] ), name='out')

        with tf.variable_scope('NN'):
            self.NN  = self.inp * 1
            for i, (l, a, d) in enumerate(zip(layers, activations, dropout)):
                with tf.variable_scope('layer_{:05d}'.format(i)):
                    self.NN = tf.layers.dense( self.NN, l, None, name='dense_{:05d}'.format(i))
                    self.NN = tf.layers.batch_normalization( self.NN,  training = self.isTraining)
                    if a is not None:
                        self.NN = a(self.NN)
                    self.NN = tf.layers.dropout( self.NN, rate=self.dropoutNN[i], name='dropout_{:05d}'.format( i ) )

        with tf.variable_scope('error'):
            self.err = tf.reduce_mean((self.NN - self.out)**2, name='err')

        with tf.variable_scope('optimizer'):
            #https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
            self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(self.update_ops):
                self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize( self.err )

        with tf.variable_scope('others'):

            tVars = tf.trainable_variables()

            graph = tf.get_default_graph()
            for v in graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES): 
                if all([
                        ('batch_normalization' in v.name),
                        ('optimizer' not in v.name), 
                        v not in tVars ]):
                    tVars.append(v)

            
            self.init  = tf.global_variables_initializer()
            self.saver = tf.train.Saver(var_list = tVars )

        logger.info('Successfully generated a graph')

        return

    @lD.log(logBase + '.SequentialRegress.saveModel')
    def saveModel(logger, self, sess):
        '''save model parameters
        
        Function that is used for generating a checkpoint so that
        a model can be reinitialized to the point at which it is 
        checkpointed. It will autogenerate a new path based upon
        the datetime at which the model is generated, and will 
        also update variable `restorePoints` of the instance so
        that there is a set of current paths already saved.
        
        Parameters
        ----------
        logger : {logging.Logger}
                The logger function
        self : {instance}
            the current instance of the class
        sess : {tf.Session() instance}
            Pass an instance of the session that you are working
            on.
        
        
        Returns
        -------
        str
            path to the file that contains the model
        '''

        path = None
        try:

            now = dt.now().strftime('%Y-%m-%d--%H-%M-%S')
            modelFolder = '../models/model-{}-{}'.format(self.name, now)
            os.makedirs(modelFolder)

            path = self.saver.save( sess, os.path.join( modelFolder, 'model.ckpt' ) )
            self.restorePoints.append(path)
        except Exception as e:
            logger.error('Unable to save the model: {}'.format(e))
            return None

        return path

    @lD.log(logBase + '.SequentialRegress.loadModel')
    def loadModel(logger, self, sess, restorePoint):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {logging.Logger}
                The logger function
        self : {instance}
            the current instance of the class
        sess : {tf.Session() instance}
            Pass an instance of the session that you are working
            on.
        restorePoint : {str}
            The path where the checkpoint is saved. 
        '''


        try:
            self.saver.restore(sess, restorePoint)
        except Exception as e:
            logger.error('Unable to restore the session at [{}]:{}'.format(
                restorePoint, str(e)))

        return

    @lD.log(logBase + '.SequentialRegress.fit')
    def fit(logger, self, X, y, N=1000, show=50, restorePoint=None):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        X : {[type]}
            [description]
        y : {[type]}
            [description]
        N : {number}, optional
            [description] (the default is 1000, which [default_description])
        show : {number}, optional
            [description] (the default is 50, which [default_description])
        restorePoint : {[type]}, optional
            [description] (the default is None, which [default_description])
        
        Returns
        -------
        [type]
            [description]
        '''
        
        try:
            with tf.Session() as sess:
                sess.run(self.init)
                if restorePoint is not None:
                    self.loadModel(sess, restorePoint)

                feed_dict = {}
                for k in self.feed_dict:
                    feed_dict[k] = self.feed_dict[k]

                feed_dict['Misc/isTraining:0'] = True
                feed_dict[self.inp] = X
                feed_dict[self.out] = y

                for i in tqdm(range(N)):
                    if i % 50 == 0:
                        _, errVal = sess.run([self.opt, self.err], 
                            feed_dict=feed_dict)
                        tqdm.write('Err Value = {}'.format( errVal ))
                    else:
                        sess.run(self.opt, 
                            feed_dict=feed_dict)

                errVal = sess.run(self.err, 
                            feed_dict=feed_dict)

                print('Final error value: {}'.format( errVal ))

                path = self.saveModel(sess)

        except Exception as e:
            logger.error('Unable to train a model ...: {}'.format(e))
            return None

        return path

    @lD.log(logBase + '.SequentialRegress.predict')
    def predict(logger, self, X, restorePoint=None):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        X : {[type]}
            [description]
        restorePoint : {[type]}, optional
            [description] (the default is None, which [default_description])
        
        Returns
        -------
        [type]
            [description]
        '''

        try:

            if (restorePoint is None) and (self.restorePoints == []):
                return None

            if restorePoint is None:
                restorePoint = self.restorePoints[-1]

            with tf.Session() as sess:
                # sess.run(self.init)
                self.loadModel(sess, restorePoint)

                feed_dict = {}
                for k in self.feed_dict:
                    feed_dict[k] = self.feed_dict[k]

                feed_dict['Misc/isTraining:0'] = False
                feed_dict[self.inp] = X

                yHat = sess.run(self.NN, feed_dict=feed_dict)
                return yHat

        except Exception as e:
            logger.error('Unable to make a prediction: {}'.format(e))
            return None

        return

    @lD.log(logBase + '.SequentialRegress.calcErr')
    def calcErr(logger, self, X, y, restorePoint=None):
        '''[summary]
        
        [description]
        
        Parameters
        ----------
        logger : {[type]}
            [description]
        self : {[type]}
            [description]
        X : {[type]}
            [description]
        restorePoint : {[type]}, optional
            [description] (the default is None, which [default_description])
        
        Returns
        -------
        [type]
            [description]
        '''

        try:

            if (restorePoint is None) and (self.restorePoints == []):
                return None

            if restorePoint is None:
                restorePoint = self.restorePoints[-1]

            with tf.Session() as sess:
                # sess.run(self.init)
                self.loadModel(sess, restorePoint)

                feed_dict = {}
                for k in self.feed_dict:
                    feed_dict[k] = self.feed_dict[k]

                feed_dict['Misc/isTraining:0'] = False
                feed_dict[self.inp] = X
                feed_dict[self.out] = y

                yHat = sess.run(self.err, feed_dict=feed_dict)
                return yHat

        except Exception as e:
            logger.error('Unable to make a prediction: {}'.format(e))
            return None

        return

