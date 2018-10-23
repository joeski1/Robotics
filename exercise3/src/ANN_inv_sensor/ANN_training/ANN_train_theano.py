#!/usr/bin/env python
# Code to train and optimise the neural network
# Optimisation will be carried out via Dropout and early stopping point
# Author: Charlie Street, Matthew Broadway

# try
# sudo apt-get install libatlas-base-dev
# then following the scipy docs to build numpy
# python setup.py build_ext --inplace to build such that it can be called from the directory
# https://docs.scipy.org/doc/numpy-1.10.4/user/install.html
# https://hunseblog.wordpress.com/2014/09/15/installing-numpy-and-openblas/

# for gpu backend in the labs
# module load cuda
# setenv THEANO_FLAGS device=gpu,force_device=True,mode=FAST_RUN
#
# for multiple threads, check that it makes a difference with:
# OMP_NUM_THREADS=1 python /usr/local/lib/python2.7/dist-packages/theano/misc/check_blas.py -q
# OMP_NUM_THREADS=2 python /usr/local/lib/python2.7/dist-packages/theano/misc/check_blas.py -q
#
# then:
# OMP_NUM_THREADS=4 python ANN_train_theano.py

import sys
import numpy as np  # used for mathematical calculations
import matplotlib  # used for plotting results from tests
import csv  # used for reading in training data
import theano
import theano.tensor as T # for machine learning optimised function generation
from theano.tensor.shared_randomstreams import RandomStreams
import time

DEBUG_THEANO = False
if DEBUG_THEANO:
    print('RUNNING THEANO IN DEBUG MODE')
    theano.config.optimizer = 'None'
    theano.config.exception_verbosity = 'high'
else:
    theano.config.allow_gc = False # re-use buffers between calls

    # fast run means spend more time optimising during compilation
    theano.config.mode = 'FAST_RUN'
    theano.config.optimizer = 'fast_run'
    theano.config.cxxflags = "-O3 -march=native -mtune=native"
    theano.config.fastmath = True

    # debugging
    #theano.config.optimizer='None' # gives better debugging info
    #theano.config.profile = True
    #theano.config.profile_memory = True
    #theano.config.exception_verbosity='high'


# parameters to tune through various optimisation methods
NUM_INPUT_NEURONS = 6
NUM_HIDDEN_NEURONS = 15
DROPOUT_RATE = 0.8
RUN_DROPOUT = True
K_FOLDS_VALUE = 15
K_FOLDS_REPEATS = 100
ETA = 0.1
EPOCHS = 100

# file with all of the training data contained
TRAINING_FILE = 'training_data.csv'

float_type = np.float32
theano.config.floatX = 'float32'

# data from training set
class TrainingSet:
    __slots__ = ('X', 't')
    def __init__(self, X=None, t=None, copy=True):
        if X is not None and copy:
            self.X = np.matrix(X, dtype=float_type, copy=True)
        else:
            self.X = X

        if t is not None and copy:
            self.t = np.matrix(t, dtype=float_type, copy=True)
        else:
            self.t = t

    def copy(self):
        return TrainingSet(self.X, self.t, copy=True)

    def create_spoof(self, num_samples=10000):
        # create lists to start with
        Xl = [(0.45,0.5, 0.7, 0.3, 0.8, 0.3)] * num_samples
        tl = [1] * num_samples

        self.X = np.matrix(Xl, dtype=float_type)
        self.t = np.matrix(tl, dtype=float_type).transpose()


    def from_csv(filename):
        '''
        converts data from a csv file into a matrix form for calculation with the neural network
        '''
        with open(filename, 'rb') as train_file:
            test = '1, 2, 3, 4, 5, 6, 10\n2, 4, 6, 8, 12, 14, 20\n'
            training_array = np.loadtxt(test, delimiter=',', dtype=float_type)
            X_arr, t_arr = np.hsplit(training_array, [6]) # split at index 6

        X = np.matrix(X, dtype=float_type) # converting to numpy matrix as faster (and more space efficient)
        t = np.matrix(t.transpose(), dtype=float_type) # numpy is significantly faster according to tests I found on stack overflow


# weights for neural network
# initialise our weight vectors randomly
class Weights:
    __slots__ = ('in_hidden', 'hidden_out')

    def __init__(self, other_weights=None):
        if other_weights is None:
            self.randomize()
        else:
            self.in_hidden = np.matrix(other_weights.in_hidden, dtype=float_type, copy=True)
            self.hidden_out = np.matrix(other_weights.hidden_out, dtype=float_type, copy=True)

    def randomize(self):
        # randn draws from a normal distribution centered at 0. The arguments
        # are the dimensions
        self.in_hidden = np.matrix(
                np.random.randn(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS),
                copy=True, dtype=float_type)
        self.hidden_out = np.matrix(
                np.random.randn(NUM_HIDDEN_NEURONS),
                copy=True, dtype=float_type)

    def copy(self):
        return Weights(self)



def get_k_fold_set(training):
    '''take the full training set with target labels
       and split up into training and testing set
       the k value defines the size of the testing set with regards to the training set
       :return (training X, training t, testing X, testing t)
    '''
    global K_FOLDS_VALUE

    ''' OLD VERSION
    X_test = []
    t_test = []
    X_train = []
    t_train = []

    size_of_X = np.size(X[:,0])
    indexes = np.arange(size_of_X)
    np.random.shuffle(indexes)  # shuffle all the indexes about
    size_of_fold = size_of_X//K_FOLDS_VALUE
    for i in range(np.size(indexes)):
        p = indexes[i]
        if i < size_of_fold: # put into test set
            X_test.append(X[p,:])
            t_test.append(t[p])
        else:
            X_train.append(X[p,:])
            t_train.append(t[p])
    X_test = np.asmatrix(np.array(X_test))
    t_test = np.asmatrix(np.array(t_test))
    X_train = np.asmatrix(np.array(X_train))
    t_train = np.asmatrix(np.array(t_train))
    '''

    set_with_labels = np.concatenate((training.X,training.t),axis=1)  # should consist of rows of (x_i, t_i)
    np.random.shuffle(set_with_labels)  # shuffle them about a bit to have random folds
    size_of_fold = np.size(training.X[:,0])//K_FOLDS_VALUE  # get the size of the fold (require integer division)
    X_test = set_with_labels[0:size_of_fold,0:6]
    t_test = set_with_labels[0:size_of_fold,6]
    X_train = set_with_labels[size_of_fold:,0:6]
    t_train = set_with_labels[size_of_fold:,6]


    return X_train, t_train, X_test, t_test


def sigmoid(x):
    '''
    sigmoid/logistic function, output function for network
    '''
    return 1/(1 + np.exp(-x))


def cross_entropy(t_p, y_p):
    '''
    cross entropy loss function for neural network
    :param t_p: true value
    :param y_p: predicted value
    :return: cross entropy loss between true value and predicted value
    '''
    return -1 * (np.multiply(t_p, np.log(y_p)) + np.multiply((1-t_p), np.log(1-y_p)))


def forward_step(x_i, weights):
    '''
    carries out forward step of neural network
    :param x_i: input to neural network
    :return: the predicted output y_p from the neural network
    '''

    signals_to_hidden = weights.in_hidden * x_i.T  # do linear combination in one line for input to hidden
    out_of_hidden = np.tanh(signals_to_hidden)
    y_p = sigmoid((weights.hidden_out * out_of_hidden)[0, 0])  # linear combination from hidden to output layer; apply sigmoid
    return y_p

def new_epoch(epoch):
    #print('E ', end='')
    if (epoch+1) % 10 == 0:
        print 'E {}'.format(epoch+1)
    else:
        print 'E',
        sys.stdout.flush()

def back_propagate(train, weights):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    for epoch in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        new_epoch(epoch)

        sample_order = np.arange(np.size(train.X[:,0]))
        np.random.shuffle(sample_order)  # randomise the order we get our training samples from
        for i in range(np.size(train.X[:, 0])):
            p = sample_order[i]
            x_p = np.asmatrix(train.X[p, :])  # training sample
            t_p = train.t[p]  # ground truth

            # forward step
            s_l1 = weights.in_hidden * x_p.T  # calculate signals going into hidden neurons
            a_l1 = np.tanh(s_l1)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial(n=1, p=DROPOUT_RATE, size=(NUM_HIDDEN_NEURONS, 1))
                a_l1 = np.multiply(a_l1, selected_neurons)  # actually make the change to temporarily remove some neurons

            s_l2 = (weights.hidden_out * a_l1)[0, 0]
            y_p = sigmoid(s_l2) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            hidden_out_weight_update = -1 * ETA * a_l1 * delta_l2  # write this properly
            weights.hidden_out += hidden_out_weight_update.T  # apply weight update

            # weight update for layer 1 (hidden layer)
            one_minus_square = np.vectorize(lambda x: 1-x**2)
            one_minus_square_a_l1 = one_minus_square(a_l1)
            delta_l1 = delta_l2 * np.multiply(one_minus_square_a_l1.T , weights.hidden_out)
            in_hidden_weight_update = -1 * ETA * delta_l1.T * x_p
            #print_dimensions(in_hidden_weight_update, weights.in_hidden, weights.in_hidden)
            weights.in_hidden += in_hidden_weight_update



def float_mat(m):
    ''' convert a matrix to have the correct representation for back propagation '''
    return np.asmatrix(np.ndarray.astype(m, dtype=float_type, copy=True, order='C'))

def tmp_mat(rows, cols):
    ''' create a matrix for holding temporary results '''
    return np.matrix(np.empty((rows, cols), dtype=float_type, order='C'),
            dtype=float_type, copy=False)

def print_dimensions(ma, mb, mc):
    print('ma = ', ma.shape)
    print('mb = ', mb.shape)
    print('mc = ', mc.shape)

# faster than the default of using a dict for object attributes
class NNLayer:
    __slots__ = ('s', 'a')
    def __str__(self):
        print('NNLayer(s={}, a={})'.format(self.s, self.a))

def back_propagate_fast(train, weights):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    # allocate variables to hold temporary results
    num_samples = train.X.shape[0] # num rows
    xs = [np.matrix(train.X[p, :], copy=False) for p in range(num_samples)]

    samples = np.array([(xs[p], xs[p].T, train.t.item(p,0)) for p in range(num_samples)])

    l1 = NNLayer()
    l1.s = tmp_mat(weights.in_hidden.shape[0], 1)
    l1.a = l1.s # re-use l1.s since it isn't used after assignment

    l2 = NNLayer()
    l2.s = tmp_mat(weights.hidden_out.shape[0], l1.a.shape[1])
    l2.a = None

    hidden_out_weight_update = tmp_mat(*l1.a.shape)
    delta_l1 = tmp_mat(l1.a.shape[1], weights.hidden_out.shape[1]) # vector
    in_hidden_weight_update = tmp_mat(delta_l1.shape[1], train.X.shape[1])

    # transposes (makes shallow copy which shares data)
    hidden_out_weight_update_T = hidden_out_weight_update.T
    l1_a_T = l1.a.T
    delta_l1_T = delta_l1.T

    for epoch in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        new_epoch(epoch)

        np.random.shuffle(samples)  # randomise the order we get our training samples from

        for x_p, x_p_T, t_p in samples:
            # x_p is a row vector of a training sample
            # x_p_T is a column vector
            # t_p is ground truth

            # forward step
            np.dot(weights.in_hidden, x_p_T, out=l1.s) # calculate signals going into hidden neurons
            np.tanh(l1.s, out=l1.a)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial(n=1, p=DROPOUT_RATE, size=(NUM_HIDDEN_NEURONS, 1))
                np.multiply(l1.a, selected_neurons, out=l1.a)  # actually make the change to temporarily remove some neurons

            # l2.s is a 1x1 matrix
            np.dot(weights.hidden_out, l1.a, out=l2.s)
            y_p = sigmoid(l2.s.item(0)) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            # first argument is a scalar, second is a vector
            np.multiply(-1 * ETA * delta_l2, l1.a, out=hidden_out_weight_update) # write this properly
            weights.hidden_out += hidden_out_weight_update_T  # apply weight update

            # weight update for layer 1 (hidden layer)
            # modify l1.a in-place. Element wise operation: 1-x^2
            #np.power(l1.a, 2, out=l1.a)
            np.multiply(l1.a, l1.a, out=l1.a) # faster than pow
            np.subtract(1, l1.a, out=l1.a)

            np.multiply(l1_a_T, weights.hidden_out, out=delta_l1)
            delta_l1 *= delta_l2

            np.multiply(delta_l1_T, x_p, out=in_hidden_weight_update)
            in_hidden_weight_update *= -1 * ETA
            weights.in_hidden += in_hidden_weight_update



class TheanoShared:
    def __init__(self):
        self.w_in_hidden = None
        self.w_hidden_out = None
        self.X = None
        self.t = None
        self.binomial = None
        self.i = None
def generate_sample_update_theano(num_samples):
    ''' some notes about theano
        - use `some_matrix = theano.printing.Print('description')(some_matrix)` because the matrix has
          to be evaluated for it to be printed, so have to use the printed
          version later in the calculation. (this is probably the issue if it's not printing)
    '''
    if float_type == np.float32:
        print('using theano float32')
        mat_type = 'float32'
    else:
        print('using theano float64')
        mat_type = 'float64'

    weights_shape = (NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS)
    w_in_hidden  = theano.shared(np.empty(weights_shape, dtype=float_type), 'w_in_hidden', strict=True)
    w_hidden_out = theano.shared(np.empty(weights_shape, dtype=float_type), 'w_hidden_out', strict=True)
    X = theano.shared(np.empty((num_samples, NUM_INPUT_NEURONS), dtype=float_type), 'X', strict=True)
    t = theano.shared(np.empty((num_samples, 1), dtype=float_type), 't', strict=True)
    binomial = theano.shared(np.empty((NUM_HIDDEN_NEURONS, num_samples), dtype=float_type), 'binomial', strict=True)
    i_mat = theano.shared(np.empty((1, 1), dtype=np.int), 'i', strict=True)

    shared = TheanoShared()
    shared.w_in_hidden = w_in_hidden
    shared.w_hidden_out = w_hidden_out
    shared.X = X
    shared.t = t
    shared.binomial = binomial
    shared.i = i_mat

    i = i_mat.take(0) # require a scalar

    x_n = T.as_tensor_variable(X[i,:], name='x_n', ndim=2)
    t_n = t[i,:]
    #x_n = T.matrix('x_n', dtype=mat_type)
    #t_n = T.matrix('t_n', dtype=mat_type)

    l1_s = T.dot(w_in_hidden, x_n.T)
    a_l1 = T.tanh(l1_s)

    if RUN_DROPOUT:
        # select which neurons to set to 0 according to dropout rate selected
        # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
        selected_neurons = T.as_tensor_variable(binomial[:,i], name='binomial_n', ndim=2).T
        #selected_neurons = theano.printing.Print('description')(selected_neurons)
        a_l1 = a_l1 * selected_neurons # element-wise, multiplication temporarily removes some neurons


    s_l2 = T.dot(w_hidden_out, a_l1) # scalar (1x1 matrix)
    y_n = T.nnet.ultra_fast_sigmoid(s_l2)
    #y_n = T.nnet.sigmoid(s_l2)

    # weight update layer 2
    delta_l2 = (y_n - t_n).take(0) # convert to scalar
    hidden_out_weight_update = (-1*ETA * a_l1 * delta_l2).T
    w_hidden_out_prime = w_hidden_out + hidden_out_weight_update

    # weight update layer 1 (hidden layer)
    one_minus_sq_l1 = 1 - (a_l1 * a_l1) # element-wise
    delta_l1 = delta_l2 * (one_minus_sq_l1.T * w_hidden_out_prime)
    in_hidden_weight_update = -1*ETA * T.dot(delta_l1.T, x_n)
    w_in_hidden_prime = w_in_hidden + in_hidden_weight_update

    i_prime = i_mat + 1

    return theano.function(
            inputs=[],
            updates=[(w_in_hidden, w_in_hidden_prime),
                     (w_hidden_out, w_hidden_out_prime),
                     (i_mat, i_prime)]
    ), shared


theano_update, theano_shared = generate_sample_update_theano(10000)

def back_propagate_theano(train, weights):

    num_samples = train.X.shape[0] # num rows

    samples = np.hstack((train.X, train.t)) # glue side by side

    theano_shared.w_in_hidden.set_value(np.asarray(weights.in_hidden))
    theano_shared.w_hidden_out.set_value(np.asarray(weights.hidden_out))

    for epoch in range(EPOCHS):
        new_epoch(epoch)

        binomial = np.random.binomial(n=1, p=DROPOUT_RATE, size=(NUM_HIDDEN_NEURONS, num_samples))
        theano_shared.binomial.set_value(binomial.astype(float_type))

        np.random.shuffle(samples)
        theano_shared.X.set_value(np.asarray(samples[:,:-1])) # all but last column
        theano_shared.t.set_value(np.asarray(samples[:,-1])) # last column
        theano_shared.i.set_value(np.array([[0]]))

        for i in range(num_samples):
            #x_n, t_n = train.X[i,:], train.t[i,:]
            theano_update()

    weights.in_hidden = theano_shared.w_in_hidden.get_value()
    weights.hidden_out = theano_shared.w_hidden_out.get_value()



def get_avg_kfold_err(training,weights):
    '''
    gets avg kfold error for a network trained with a generated subset of the full training data,
             with multiple repeats
    :return: avg kfold error
    '''
    total_sum = 0.0  # will store final result
    for _ in range(K_FOLDS_REPEATS):  # do multiple repeats to reduce validation bias
        local_sum = 0.0
        for _ in range(K_FOLDS_VALUE):  # do for each of the k folds
            fold_sum = 0.0
            X_train, t_train, X_test, t_test = get_k_fold_set(training)  # get the partition

            weights.randomize()  # reinitialise weights to stop it using weights from previously
            back_propagate(TrainingSet(X_train, t_train, copy=False),weights)  # train the network with the training set partition

            # get total error for this fold
            for i in range(np.size(X_test[:, 0])):
                fold_sum += cross_entropy(t_test[i], forward_step(X_test[i, :]))
            local_sum += fold_sum  # add total fold error to local sum, this will be averaged over k folds later

        total_sum += (local_sum / float(K_FOLDS_VALUE))
    return total_sum / float(K_FOLDS_REPEATS)


def get_avg_training_err(train):
    '''
    as part of using dropout as a regularisation method we first need to over-fit our training data greatly
    in order to ensure we over-fit in this case, we can use the training error to give us some measure
    :return: the average training error over all training samples for the current weight vectors
    '''

    total_sum = 0.0
    no_samples = np.size(train.X[:,0])
    for p in range(no_samples):
        total_sum += cross_entropy(train.t[p],forward_step(train.X[p,:]))  # loss for a given sample

    return total_sum / float(no_samples)  # get average of all losses for individual training samples

# backups
train = TrainingSet()
train.create_spoof()
w = Weights()

EPOCHS = 10
import time

# slow
print('timing back propagation')
s_tr = train.copy()
s_w = Weights(w)
start = time.time()
back_propagate(s_tr, s_w)
slow_time = time.time()-start
print('\ntook {} seconds for {} epochs'.format(slow_time, EPOCHS))


# fast
print('timing fast back propagation')
f_tr = train.copy()
f_w = Weights(w)
start = time.time()
back_propagate_fast(f_tr, f_w)
fast_time = time.time()-start
print('\ntook {} seconds for {} epochs'.format(fast_time, EPOCHS))

print('timing theano back propagation')
t_tr = train.copy()
t_w = Weights(w)
start = time.time()
back_propagate_theano(t_tr, t_w)
theano_time = time.time()-start
print('\ntook {} seconds for {} epochs'.format(theano_time, EPOCHS))

print
print('speedup fast   = {}'.format(slow_time/fast_time))
print('speedup theano = {}'.format(slow_time/theano_time))

print

if float_type == np.float32:
    rtol=1e-3
else:
    rtol=1e-6

print('comparing with fast')
print('weights.in_hidden\'s: {}'.format(np.allclose(s_w.in_hidden, f_w.in_hidden, rtol=rtol)))
print('weights.hidden_out\'s: {}'.format(np.allclose(s_w.hidden_out, f_w.hidden_out, rtol=rtol)))
print('comparing with theano')
print('weights.in_hidden\'s: {}'.format(np.allclose(s_w.in_hidden, t_w.in_hidden, rtol=rtol)))
print('weights.hidden_out\'s: {}'.format(np.allclose(s_w.hidden_out, t_w.hidden_out, rtol=rtol)))

#print(f_w.in_hidden)
#print
#print(t_w.in_hidden)

#print('compare with originals')
#print('weights.hidden_out\'s: {}'.format(np.allclose(w.in_hidden, s_w.in_hidden)))
#print('weights.hidden_out\'s: {}'.format(np.allclose(w.hidden_out, s_w.hidden_out)))

#x_train,t_train,x_test,t_test = get_k_fold_set()

