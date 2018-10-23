#!/usr/bin/env python
# Code to train and optimise the neural network
# Optimisation will be carried out via Dropout and early stopping point
# Author: Charlie Street, Matthew Broadway

USE_GPU=True

# this documentation is very useful: https://www.cs.toronto.edu/~vmnih/docs/cudamat_tr.pdf
# also see the cudamat examples on github
if USE_GPU:
    import cudamat as cm
    cm.cublas_init()

import sys
import numpy as np  # used for mathematical calculations
#import matplotlib  # used for plotting results from tests
import csv  # used for reading in training data


# parameters to tune through various optimisation methods
NUM_INPUT_NEURONS = 6
NUM_HIDDEN_NEURONS = 15
DROPOUT_RATE = 0.8
RUN_DROPOUT = False
K_FOLDS_VALUE = 15
K_FOLDS_REPEATS = 100
ETA = 0.1
EPOCHS = 100

# file with all of the training data contained
TRAINING_FILE = 'training_data.csv'

# data from training set in python usable form
# start as lists but converted to numpy arrays as more efficient
X = []
t = []

# weights for neural network
# initialise our weight vectors randomly
w_in_hidden = np.asmatrix(np.random.randn(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS))
w_hidden_out = np.asmatrix(np.random.randn(NUM_HIDDEN_NEURONS))


def training_spoof_matrix():
    global X, t

    X = []
    t = []

    for i in range(10000):
        X.append((0.45,0.5, 0.7, 0.3, 0.8, 0.3))
        t.append(1)

    X = np.asmatrix(np.array(X))
    t = np.asmatrix(np.array(t)).T


def training_to_matrix():
    '''
    function converts all data from csv file into a matrix form for calculation with the neural network
    '''

    global X, t

    with open(TRAINING_FILE, 'rb') as train_file:  # open up our file
        file_reader = csv.reader(train_file)
        for row in file_reader:
            x_i = (float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            X.append(x_i)
            t.append(float(row[6]))

    X = np.asmatrix(np.array(X))  # converting to numpy matrix as faster (and more space efficient)
    t = np.asmatrix(np.array(t)).T  # numpy is significantly faster according to tests I found on stack overflow


def reinit_weights():
    '''
    reinitialise the random weights for future training sample
    :return: nothing; update the weights to be random again
    '''
    global w_in_hidden, w_hidden_out

    w_in_hidden = np.asmatrix(np.random.randn(NUM_HIDDEN_NEURONS, NUM_INPUT_NEURONS))
    w_hidden_out = np.asmatrix(np.random.randn(NUM_HIDDEN_NEURONS))


def get_k_fold_set():
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

    set_with_labels = np.concatenate((X,t),axis=1)  # should consist of rows of (x_i, t_i)
    np.random.shuffle(set_with_labels)  # shuffle them about a bit to have random folds
    size_of_fold = np.size(X[:,0])//K_FOLDS_VALUE  # get the size of the fold (require integer division)
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


def cross_entropy(t_p,y_p):
    '''
    cross entropy loss function for neural network
    :param t_p: true value
    :param y_p: predicted value
    :return: cross entropy loss between true value and predicted value
    '''
    return -1 * (np.multiply(t_p, np.log(y_p)) + np.multiply((1-t_p), np.log(1-y_p)))


def forward_step(x_i):
    '''
    carries out forward step of neural network
    :param x_i: input to neural network
    :return: the predicted output y_p from the neural network
    '''

    global w_in_hidden, w_hidden_out

    signals_to_hidden = w_in_hidden * x_i.T  # do linear combination in one line for input to hidden
    out_of_hidden = np.tanh(signals_to_hidden)
    y_p = sigmoid((w_hidden_out * out_of_hidden)[0, 0])  # linear combination from hidden to output layer; apply sigmoid
    return y_p


def back_propagate(X_train,t_train):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    global w_hidden_out, w_in_hidden, EPOCHS, ETA, DROPOUT_RATE, RUN_DROPOUT

    for _ in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        print 'E',
        sys.stdout.flush()

        sample_order = np.arange(np.size(X_train[:,0]))
        np.random.shuffle(sample_order)  # randomise the order we get our training samples from
        for i in range(np.size(X_train[:, 0])):
            p = sample_order[i]
            x_p = np.asmatrix(X_train[p, :])  # training sample
            t_p = t_train[p]  # ground truth

            # forward step
            s_l1 = w_in_hidden * x_p.T  # calculate signals going into hidden neurons
            a_l1 = np.tanh(s_l1)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial([np.ones(NUM_HIDDEN_NEURONS)],DROPOUT_RATE)[0]
                a_l1 = np.multiply(a_l1,selected_neurons)  # actually make the change to temporarily remove some neurons

            s_l2 = (w_hidden_out * a_l1)[0, 0]
            y_p = sigmoid(s_l2) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            hidden_out_weight_update = -1 * ETA * a_l1 * delta_l2  # write this properly
            w_hidden_out += hidden_out_weight_update.T  # apply weight update

            # weight update for layer 1 (hidden layer)
            one_minus_square = np.vectorize(lambda x: 1-x**2)
            one_minus_square_a_l1 = one_minus_square(a_l1)
            delta_l1 = delta_l2 * np.multiply(one_minus_square_a_l1.T , w_hidden_out)
            in_hidden_weight_update = -1 * ETA * delta_l1.T * x_p
            w_in_hidden += in_hidden_weight_update



mat_type = np.float64
def float_mat(m):
    ''' convert a matrix to have the correct representation for back propagation '''
    return np.asmatrix(np.ndarray.astype(m, dtype=mat_type, order='C', copy=True))

def tmp_mat(rows, cols):
    ''' create a matrix for holding temporary results '''
    return np.matrix(np.empty((rows, cols), dtype=mat_type, order='C'),
            dtype=mat_type, copy=False)

def print_dimensions(ma, mb, mc):
    print('ma = ', ma.shape)
    print('mb = ', mb.shape)
    print('mc = ', mc.shape)

# faster than the default of using a dict for object attributes
class NNLayer:
    __slots__ = ('s', 'a')
    def __str__(self):
        print('NNLayer(s={}, a={})'.format(self.s, self.a))

def back_propagate_fast(X_train,t_train):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    global w_hidden_out, w_in_hidden, EPOCHS, ETA, DROPOUT_RATE, RUN_DROPOUT

    # allocate variables to hold temporary results
    l1 = NNLayer()
    l1.s = tmp_mat(w_in_hidden.shape[0], 1)
    l1.a = tmp_mat(*l1.s.shape) # same shape as s_l1

    l2 = NNLayer()
    l2.s = tmp_mat(w_hidden_out.shape[0], l1.a.shape[1])
    l2.a = None

    hidden_out_weight_update = tmp_mat(*l1.a.shape)
    delta_l1 = tmp_mat(l1.a.shape[1], w_hidden_out.shape[1]) # vector
    in_hidden_weight_update = tmp_mat(delta_l1.shape[1], X_train.shape[1])

    for _ in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        print 'E',
        sys.stdout.flush()

        num_samples = X_train.shape[0] # num rows
        sample_order = np.arange(num_samples)
        np.random.shuffle(sample_order)  # randomise the order we get our training samples from

        for p in sample_order:
            x_p = np.matrix(X_train[p, :], copy=False)  # training sample (row vector)
            t_p = t_train.item(p, 0)  # ground truth (scalar)

            # forward step
            np.dot(w_in_hidden, x_p.T, out=l1.s) # calculate signals going into hidden neurons
            np.tanh(l1.s, out=l1.a)

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial([np.ones(NUM_HIDDEN_NEURONS)], DROPOUT_RATE)[0]
                l1.a = np.multiply(l1.a, selected_neurons)  # actually make the change to temporarily remove some neurons

            # l2.s is a 1x1 matrix
            np.dot(w_hidden_out, l1.a, out=l2.s)
            y_p = sigmoid(l2.s.item(0)) # get the predicted value from the network

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            # first argument is a scalar, second is a vector
            np.multiply(-1 * ETA * delta_l2, l1.a, out=hidden_out_weight_update) # write this properly
            w_hidden_out += hidden_out_weight_update.T  # apply weight update

            # weight update for layer 1 (hidden layer)
            # modify l1.a in-place. Element wise operation: 1-x^2
            np.power(l1.a, 2, out=l1.a)
            np.subtract(1, l1.a, out=l1.a)

            np.multiply(l1.a.T, w_hidden_out, out=delta_l1)
            delta_l1 *= delta_l2

            np.dot(delta_l1.T, x_p, out=in_hidden_weight_update)
            in_hidden_weight_update *= -1 * ETA
            w_in_hidden += in_hidden_weight_update


def tmp_mat_gpu(rows, cols):
    ''' create a matrix for holding temporary results '''
    mat = cm.empty((rows, cols))
    mat.copy_to_host() # this ensures that the attribute numpy_array is allocated
    return mat
def to_gpu_mat(cpu_mat):
    converted_cpu = np.asarray(cpu_mat, dtype=np.float32, order='F')
    return cm.CUDAMatrix(converted_cpu)
def back_propagate_gpu(X_train, t_train):
    '''
    carries out back propagation on a neural network
    this is for a neural network used for binary classification
    :return: nothing. The return is the trained neural network
    '''

    global w_hidden_out, w_in_hidden, EPOCHS, ETA, DROPOUT_RATE, RUN_DROPOUT

    w_hidden_out = to_gpu_mat(w_hidden_out)
    w_in_hidden = to_gpu_mat(w_in_hidden)

    # allocate variables to hold temporary results
    l1 = NNLayer()
    l1.s = tmp_mat_gpu(w_in_hidden.shape[0], 1)
    l1.a = tmp_mat_gpu(*l1.s.shape) # same shape as s_l1
    l1_a_T = tmp_mat_gpu(l1.s.shape[1], l1.s.shape[0])

    l2 = NNLayer()
    l2.s = tmp_mat_gpu(w_hidden_out.shape[0], l1.a.shape[1])
    l2.a = None

    hidden_out_weight_update = tmp_mat_gpu(l1.a.shape[1], l1.a.shape[0])

    delta_l1 = tmp_mat_gpu(l1.a.shape[1], w_hidden_out.shape[1]) # vector
    in_hidden_weight_update = tmp_mat_gpu(delta_l1.shape[1], X_train.shape[1])

    xs = [to_gpu_mat(X_train[p, :]) for p in range(X_train.shape[0])]

    for _ in range(EPOCHS):  # have to iterate over the back propagation algorithm multiple times
        print 'E',
        sys.stdout.flush()

        num_samples = X_train.shape[0] # num rows
        sample_order = np.arange(num_samples)
        np.random.shuffle(sample_order)  # randomise the order we get our training samples from


        last = [0]
        d = dict()
        def tp(desc):
            eps = time.time()-last[0]
            if desc not in d:
                d[desc] = eps
            else:
                d[desc] += eps
            last[0] = time.time()

        for p in sample_order:
            last[0] = time.time()
            x_p = xs[p]  # training sample (row vector)
            t_p = t_train.item(p, 0)  # ground truth (scalar)
            tp('created x_p')

            # forward step
            cm.dot(w_in_hidden, x_p.T, target=l1.s) # calculate signals going into hidden neurons
            cm.tanh(l1.s)
            tp('forward step')

            # carry out dropout, if we have opted to
            if RUN_DROPOUT:
                # select which neurons to set to 0 according to dropout rate selected
                # appropriate credit to be given to Shan He for demonstrating this method of implementing dropout
                selected_neurons = np.random.binomial([np.ones(NUM_HIDDEN_NEURONS)], DROPOUT_RATE)[0]
                l1.a.mult(selected_neurons)  # actually make the change to temporarily remove some neurons

            # l2.s is a 1x1 matrix
            cm.dot(w_hidden_out, l1.a, target=l2.s)
            y_p = sigmoid(l2.s.asarray().item(0)) # get the predicted value from the network
            tp('done sigmoid')

            # weight update for layer 2
            delta_l2 = y_p - t_p  # I have derived this for binary classification
            # first argument is a scalar, second is a vector

            l1.a.transpose(target=hidden_out_weight_update)
            hidden_out_weight_update.mult(-1 * ETA * delta_l2) # write this properly
            # apply weight update
            w_hidden_out.add(hidden_out_weight_update)
            tp('layer 2 weight update')

            # weight update for layer 1 (hidden layer)
            # modify l1.a in-place. Element wise operation: 1-x^2
            cm.pow(l1.a, 2, target=l1.a)
            l1.a.subtract(1)
            tp('layer 1 weight update')

            l1.a.transpose(target=l1_a_T)
            w_hidden_out.mult(l1_a_T, target=delta_l1) # element-wise
            delta_l1.mult(delta_l2)

            cm.dot(delta_l1.T, x_p, target=in_hidden_weight_update)
            in_hidden_weight_update.mult(-1*ETA)

            w_in_hidden.add(in_hidden_weight_update)
            tp('done')

        for k in d:
            print('{}: {}'.format(k, d[k]))

    # copy from gpu
    w_in_hidden = w_in_hidden.asarray()
    w_hidden_out = w_hidden_out.asarray()

def get_avg_kfold_err():
    '''
    gets avg kfold error for a network trained with a generated subset of the full training data,
             with multiple repeats
    :return: avg kfold error
    '''
    global K_FOLDS_REPEATS, K_FOLDS_VALUE
    total_sum = 0.0  # will store final result
    for _ in range(K_FOLDS_REPEATS):  # do multiple repeats to reduce validation bias
        local_sum = 0.0
        for _ in range(K_FOLDS_VALUE):  # do for each of the k folds
            fold_sum = 0.0
            X_train, t_train, X_test, t_test = get_k_fold_set()  # get the partition

            reinit_weights()  # reinitialise weights to stop it using weights from previously
            back_propagate(X_train, t_train)  # train the network with the training set partition

            # get total error for this fold
            for i in range(np.size(X_test[:, 0])):
                fold_sum += cross_entropy(t_test[i], forward_step(X_test[i, :]))
            local_sum += fold_sum  # add total fold error to local sum, this will be averaged over k folds later

        total_sum += (local_sum / float(K_FOLDS_VALUE))
    return total_sum / float(K_FOLDS_REPEATS)


def get_avg_training_err():
    '''
    as part of using dropout as a regularisation method we first need to over-fit our training data greatly
    in order to ensure we over-fit in this case, we can use the training error to give us some measure
    :return: the average training error over all training samples for the current weight vectors
    '''
    global X, t

    total_sum = 0.0
    no_samples = np.size(X[:,0])
    for p in range(no_samples):
        total_sum += cross_entropy(t[p],forward_step(X[p,:]))  # loss for a given sample

    return total_sum / float(no_samples)  # get average of all losses for individual training samples

training_spoof_matrix()
# backups
b_w_in_hidden = np.matrix(w_in_hidden, copy=True)
b_w_hidden_out = np.matrix(w_hidden_out, copy=True)
b_X = np.matrix(X, copy=True)
b_t = np.matrix(t, copy=True)

EPOCHS = 100
import time

# slow backprop
print('timing back propagation')
start = time.clock()
back_propagate_fast(X,t)
slow_time = time.clock()-start
print('\ntook {} seconds for {} epochs'.format(slow_time, EPOCHS))

# slow output
s_w_in_hidden = np.matrix(w_in_hidden, copy=True)
s_w_hidden_out = np.matrix(w_hidden_out, copy=True)
s_X = np.matrix(X, copy=True)
s_t = np.matrix(t, copy=True)


# fast backprop
print('timing fast back propagation')
training_spoof_matrix()

w_in_hidden = float_mat(b_w_in_hidden)
w_hidden_out = float_mat(b_w_hidden_out)
X = float_mat(b_X)
t = float_mat(b_t)


start = time.clock()
#back_propagate_fast(X,t)
back_propagate_gpu(X,t)
fast_time = time.clock()-start
print('\ntook {} seconds for {} epochs'.format(fast_time, EPOCHS))

if fast_time != 0.0:
    print('speedup = {}'.format(slow_time/fast_time))
else:
    print('cannot calculate speedup')

print 'X\'s: ', np.allclose(s_X, X)
print 't\'s: ', np.allclose(s_t, t)
print 'w_in_hidden\'s: ', np.allclose(s_w_in_hidden, w_in_hidden)
print 'w_hidden_out\'s: ', np.allclose(s_w_hidden_out, w_hidden_out)
print(s_w_in_hidden)
print()
print(w_in_hidden)

#x_train,t_train,x_test,t_test = get_k_fold_set()

if USE_GPU:
    cm.cublas_shutdown()
